import polars as pl
import numpy as np
import pandas as pd  # 仅用于生成模拟数据展示，核心逻辑全用 Polars

class TopLongShortPositionRatioFactorEngine:
    def __init__(self, kline_df: pl.DataFrame, resample_freq='15m'):
        """
        聪明钱(Smart Money)因子挖掘引擎
        
        :param kline_df: K线上下文数据 (必须包含 'open_time', 'close')，用于计算价格背离
        :param resample_freq: 数据重采样频率 (默认15分钟)
        """
        self.freq = resample_freq
        
        # 预处理 K线数据：确保有序、类型正确
        self.kline_context = (
            kline_df.lazy()
            .select([
                pl.col("open_time").cast(pl.Datetime),
                pl.col("close").cast(pl.Float64)
            ])
            .sort("open_time")
            .collect()
        )

    def process(self, raw_whale_data: pl.DataFrame, 
                windows=[24, 96],  # 滚动窗口 (如 24*15m=6h, 96*15m=24h)
                z_window=672):     # Z-Score 回看窗口 (如 7天)
        
        print(f"[*] 启动聪明钱引擎 | 频率: {self.freq}")
        
        # 1. 数据清洗与标准化 (Data Cleaning)
        # 解决截图中的重复数据问题，并转换数值类型
        lf = self._clean_and_normalize(raw_whale_data)
        
        # 2. 上下文对齐 (Context Alignment)
        # 将大户数据与价格数据对齐，才能计算“背离”
        lf = self._align_with_price(lf)
        
        # 3. 基础特征构建 (Base Features)
        # 转换 Ratio 为 Net Position，计算一阶差分
        print("[-] 构建基础大户头寸特征...")
        lf = self._build_base_features(lf)
        
        # 4. 统计与时序特征 (Statistical & Temporal)
        # 计算 Z-Score (拥挤度) 和 ROC (变化率)
        print(f"[-] 计算统计特征 (Z-Window={z_window})...")
        lf = self._build_stat_features(lf, z_window)
        
        # 5. 猎人与猎物背离特征 (Hunter-Prey Divergence)
        # 核心逻辑：计算 Price Rank - Whale Rank
        print(f"[-] 挖掘核心背离因子 (Windows={windows})...")
        lf = self._mining_divergence(lf, windows)
        
        # 执行计算
        df_final = lf.collect()
        
        print(f"[+] 处理完成. 最终数据形状: {df_final.shape}")
        return df_final

    def _clean_and_normalize(self, df: pl.DataFrame):
        """
        清洗阶段：去重、类型转换、时间排序
        """
        return (
            df.lazy()
            # 1. 针对截图中的重复数据进行去重，保留最新的一条
            .unique(subset=["open_time", "symbol"], keep="last")
            # 2. 类型强制转换
            .with_columns([
                pl.col("open_time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
                pl.col("longShortRatio").cast(pl.Float64),
                pl.col("longAccount").cast(pl.Float64),
                pl.col("shortAccount").cast(pl.Float64),
            ])
            .sort("open_time")
            # 3. 按频率重采样 (防止数据过于密集或稀疏)
            .group_by_dynamic("open_time", every=self.freq)
            .agg([
                pl.col("longShortRatio").last(),
                pl.col("longAccount").last(),
                pl.col("shortAccount").last(),
            ])
        )

    def _align_with_price(self, lf: pl.LazyFrame):
        """
        使用 join_asof 毫秒级对齐 K 线收盘价
        """
        return lf.join_asof(
            self.kline_context.lazy(), 
            on="open_time", 
            strategy="backward"
        )

    def _build_base_features(self, lf: pl.LazyFrame):
        """
        将原始 Ratio 转化为更有物理意义的 Net Position
        """
        return lf.with_columns([
            # 1. 净头寸 (Net Position): 范围 [-1, 1]
            # 正数代表大户净看多，负数代表净看空
            (pl.col("longAccount") - pl.col("shortAccount")).alias("whale_net_pos"),
            
            # 2. 净头寸流向 (Flow): 大户是在加仓还是减仓？
            (pl.col("longAccount") - pl.col("shortAccount")).diff().alias("whale_net_flow"),
        ])

    def _build_stat_features(self, lf: pl.LazyFrame, window_size):
        """
        计算 Z-Score 以衡量“拥挤度”和“极端情绪”
        """
        # 辅助函数：计算 Z-Score
        def z_score(col_name, w):
            return (
                (pl.col(col_name) - pl.col(col_name).rolling_mean(w)) / 
                (pl.col(col_name).rolling_std(w) + 1e-9)
            ).alias(f"feat_z_{col_name}_{w}")

        return lf.with_columns([
            z_score("whale_net_pos", window_size),   # 头寸的极端程度
            z_score("longShortRatio", window_size)   # 比率的极端程度
        ])

    def _mining_divergence(self, lf: pl.LazyFrame, windows):
        """
        【核心逻辑】计算价格与大户持仓的背离
        方法论：归一化后的 Price - 归一化后的 Whale_Pos
        """
        exprs = []
        
        for w in windows:
            # 1. 价格的局部位置 (0~1)
            # (Close - Min) / (Max - Min)
            price_norm = (
                (pl.col("close") - pl.col("close").rolling_min(w)) / 
                (pl.col("close").rolling_max(w) - pl.col("close").rolling_min(w) + 1e-9)
            )
            
            # 2. 大户持仓的局部位置 (0~1)
            whale_norm = (
                (pl.col("whale_net_pos") - pl.col("whale_net_pos").rolling_min(w)) / 
                (pl.col("whale_net_pos").rolling_max(w) - pl.col("whale_net_pos").rolling_min(w) + 1e-9)
            )
            
            # 3. 背离因子 (Divergence)
            # 值 > 0.5 (正极大): 价格在高位，大户在低位 => 顶背离 (诱多/出货) -> 看空
            # 值 < -0.5 (负极大): 价格在低位，大户在高位 => 底背离 (吸筹/接盘) -> 看多
            div_name = f"feat_div_price_whale_w{w}"
            exprs.append((price_norm - whale_norm).alias(div_name))
            
            # 4. 衍生逻辑：吸筹信号 (Accumulation Signal)
            # 定义：价格低位 (Price < 0.2) 且 大户高位 (Whale > 0.8)
            signal_name = f"sig_accumulation_w{w}"
            exprs.append(
                ((price_norm < 0.2) & (whale_norm > 0.8)).cast(pl.Int8).alias(signal_name)
            )

        return lf.with_columns(exprs)

# ==========================================
# 模拟运行与测试 (Simulation)
# ==========================================
if __name__ == "__main__":
    # 1. 生成模拟 K线数据 (价格走势)
    print("生成 K 线上下文...")
    dates = pd.date_range('2025-01-01', periods=1000, freq='15min')
    # 模拟一个先跌后涨的行情，用于测试底背离
    price_trend = np.concatenate([
        np.linspace(100, 80, 500), # 下跌
        np.linspace(80, 120, 500)  # 上涨
    ]) + np.random.normal(0, 1, 1000)
    
    kline_df = pl.DataFrame({
        "open_time": dates,
        "close": price_trend
    })

    # 2. 生成模拟大户数据 (带噪音和重复)
    print("生成大户持仓数据 (包含重复项和背离逻辑)...")
    # 在价格下跌时(前500)，让大户持仓悄悄上升 (模拟吸筹背离)
    whale_pos = np.concatenate([
        np.linspace(0.4, 0.7, 500), # 价格跌，但我买
        np.linspace(0.7, 0.5, 500)  # 价格涨，我出货
    ]) + np.random.normal(0, 0.05, 1000)
    
    # 构造原始 DataFrame 结构
    raw_whale_pd = pd.DataFrame({
        "open_time": dates, # 简单起见一一对应，实际会有缺失
        "symbol": "BTCUSDT",
        "longShortRatio": whale_pos / (1 - whale_pos), # 倒推 Ratio
        "longAccount": whale_pos,
        "shortAccount": 1 - whale_pos
    })
    
    # 人为制造重复数据 (模拟你的截图情况)
    raw_whale_pd = pd.concat([raw_whale_pd, raw_whale_pd.iloc[100:200]]).sample(frac=1).reset_index(drop=True)
    # 转换时间为字符串以模拟原始 CSV 格式
    raw_whale_pd['open_time'] = raw_whale_pd['open_time'].astype(str)
    
    # 转为 Polars
    whale_pl = pl.from_pandas(raw_whale_pd)

    # 3. 运行引擎
    engine = TopLongShortPositionRatioFactorEngine(kline_df, resample_freq='15m')
    df_result = engine.process(whale_pl, windows=[96], z_window=200)

    # 4. 结果验证
    print("\n[因子预览]")
    # 选取特定时间段查看背离情况 (下跌末期)
    cols = ['open_time', 'close', 'whale_net_pos', 'feat_div_price_whale_w96', 'sig_accumulation_w96']
    preview = df_result.filter(
        (pl.col("open_time").dt.hour() == 12)  # 随便抽样
    ).select(cols).sort("open_time")
    
    print(preview.tail(10))
    
    # 解释输出
    print("\n[解读]")
    print("注意观察 'feat_div_price_whale_w96'：")
    print("如果数值接近 -1.0，说明 价格(Close)在滚动窗口的低位，而大户持仓(Whale)在高位 -> 底背离(吸筹)。")
    print("如果数值接近 +1.0，说明 价格(Close)在滚动窗口的高位，而大户持仓(Whale)在低位 -> 顶背离(出货)。")