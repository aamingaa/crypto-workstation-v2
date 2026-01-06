import polars as pl
import numpy as np
import pandas as pd  # 仅用于生成模拟数据展示，核心逻辑全用 Polars

# 大户的多头和空头总持仓量占比，大户指保证金余额排名前20%的用户。 
# 多仓持仓量比例 = 大户多仓持仓量 / 大户总持仓量 
# 空仓持仓量比例 = 大户空仓持仓量 / 大户总持仓量 
# 多空持仓量比值 = 多仓持仓量比例 / 空仓持仓量比例

# { 
#          "symbol":"BTCUSDT",
# 	      "longShortRatio":"1.4342",// 大户多空持仓量比值
# 	      "longAccount": "0.5344", // 大户多仓持仓量比例
# 	      "shortAccount":"0.4238", // 大户空仓持仓量比例
# 	      "timestamp":"1583139600000"
# }
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
        计算 Z-Score 以衡量"拥挤度"和"极端情绪"
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
    
    def _safe_normalize(self, col_expr, window, min_range=1e-6):
        """
        安全的归一化方法，处理边界情况
        
        :param col_expr: Polars 列表达式
        :param window: 滚动窗口大小
        :param min_range: 最小有效范围，低于此值视为无变化
        :return: 归一化后的表达式 (0~1)
        """
        min_val = col_expr.rolling_min(window)
        max_val = col_expr.rolling_max(window)
        range_val = max_val - min_val
        
        # 当范围过小时（价格/持仓无变化），返回中性值 0.5
        # 避免除零和数值不稳定
        return pl.when(range_val > min_range).then(
            (col_expr - min_val) / range_val
        ).otherwise(0.5)

    def _mining_divergence(self, lf: pl.LazyFrame, windows, 
                          price_low_thresh=0.2, 
                          whale_high_thresh=0.8,
                          price_high_thresh=0.8,
                          whale_low_thresh=0.2):
        """
        【核心逻辑】计算价格与大户持仓的背离
        方法论：归一化后的 Price - 归一化后的 Whale_Pos
        
        :param windows: 滚动窗口列表
        :param price_low_thresh: 价格低位阈值 (默认 0.2，即 20分位)
        :param whale_high_thresh: 大户高位阈值 (默认 0.8，即 80分位)
        :param price_high_thresh: 价格高位阈值 (默认 0.8，用于出货信号)
        :param whale_low_thresh: 大户低位阈值 (默认 0.2，用于出货信号)
        """
        exprs = []
        
        for w in windows:
            # 1. 价格的局部位置 (0~1) - 使用安全归一化
            price_norm = self._safe_normalize(pl.col("close"), w)
            
            # 2. 大户持仓的局部位置 (0~1) - 使用安全归一化
            whale_norm = self._safe_normalize(pl.col("whale_net_pos"), w)
            
            # 3. 背离因子 (Divergence)
            # 值 > 0 (正): 价格高于大户持仓 => 可能顶背离 (诱多/出货) -> 看空倾向
            # 值 < 0 (负): 价格低于大户持仓 => 可能底背离 (吸筹/接盘) -> 看多倾向
            # 值的绝对值越大，背离程度越强
            div_name = f"feat_div_price_whale_w{w}"
            exprs.append((price_norm - whale_norm).alias(div_name))
            
            # 4. 吸筹信号 (Accumulation Signal)
            # 定义：价格低位 & 大户高位 => 大户在底部建仓
            signal_accumulation = f"sig_accumulation_w{w}"
            exprs.append(
                ((price_norm < price_low_thresh) & (whale_norm > whale_high_thresh))
                .cast(pl.Int8)
                .alias(signal_accumulation)
            )
            
            # 5. 出货信号 (Distribution Signal)
            # 定义：价格高位 & 大户低位 => 大户在顶部减仓
            signal_distribution = f"sig_distribution_w{w}"
            exprs.append(
                ((price_norm > price_high_thresh) & (whale_norm < whale_low_thresh))
                .cast(pl.Int8)
                .alias(signal_distribution)
            )

        return lf.with_columns(exprs)