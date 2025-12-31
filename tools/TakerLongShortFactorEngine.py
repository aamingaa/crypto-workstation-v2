import polars as pl
import numpy as np

class TakerFlowEngine:
    def __init__(self, kline_df: pl.DataFrame, resample_freq='5m'):
        """
        :param kline_df: 必须包含 'open_time', 'close', 'high', 'low' (用于计算Effort/Result)
        """
        self.freq = resample_freq
        # 预处理 K线数据
        self.kline_context = (
            kline_df.lazy()
            .select([
                pl.col("open_time").cast(pl.Datetime),
                pl.col("close").cast(pl.Float64),
                # 计算价格变化，用于对比 Effort vs Result
                pl.col("close").diff().alias("price_change") 
            ])
            .sort("open_time")
            .collect()
        )

    def process(self, raw_taker_df: pl.DataFrame):
        print(f"[*] 启动 Taker 引擎 | 频率: {self.freq}")
        
        # 1. 清洗 (Cleaning)
        lf = self._clean_data(raw_taker_df)
        
        # 2. 对齐 K线 (Alignment)
        lf = lf.join_asof(self.kline_context.lazy(), on="open_time", strategy="backward")
        
        # 3. 基础特征 (Net Flow & Imbalance)
        lf = self._build_base_features(lf)
        
        # 4. 高级特征 (CVD Divergence & Effort/Result)
        lf = self._build_advanced_features(lf)
        
        return lf.collect()

    def _clean_data(self, df: pl.DataFrame):
        """
        清洗逻辑：
        1. 针对截图中重复的时间戳去重
        2. 处理 Unix Timestamp 或 String Date
        3. 类型转换
        """
        return (
            df.lazy()
            # [关键] 去重：针对 open_time 保留最后一条
            .unique(subset=["open_time"], keep="last")
            
            # 类型转换
            .with_columns([
                # 如果是字符串时间 "2025-10-01..."
                pl.col("open_time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
                
                # 如果是 Unix Timestamp (毫秒)，用下面这行替换上面那行:
                # pl.col("timestamp").cast(pl.Int64).cast(pl.Datetime("ms")).alias("open_time"),
                
                pl.col("buyVol").cast(pl.Float64),
                pl.col("sellVol").cast(pl.Float64),
                pl.col("buySellRatio").cast(pl.Float64)
            ])
            .sort("open_time")
            # 聚合重采样 (防止数据粒度不统一)
            .group_by_dynamic("open_time", every=self.freq)
            .agg([
                pl.col("buyVol").sum(), # Taker量通常是区间的，所以用 sum
                pl.col("sellVol").sum()
            ])
        )

    def _build_base_features(self, lf: pl.LazyFrame):
        return lf.with_columns([
            # 1. 净主动买入量 (Net Taker Volume) - 绝对值
            (pl.col("buyVol") - pl.col("sellVol")).alias("taker_net_vol"),
            
            # 2. 情绪失衡度 (Imbalance) - 归一化 [-1, 1]
            # 相比 Ratio (0~无穷)，这个对模型更友好
            ((pl.col("buyVol") - pl.col("sellVol")) / 
             (pl.col("buyVol") + pl.col("sellVol") + 1e-9)).alias("taker_imbalance_norm")
        ])

    def _build_advanced_features(self, lf: pl.LazyFrame):
        """
        挖掘 Alpha 的核心：CVD 和 努力/结果分析
        """
        return lf.with_columns([
            # 1. CVD (Cumulative Volume Delta) - 累积资金流
            pl.col("taker_net_vol").cum_sum().alias("taker_cvd"),
            
        ]).with_columns([
            # 2. CVD 斜率 (CVD Slope) - 资金流入的加速度
            pl.col("taker_cvd").diff(5).alias("feat_cvd_momentum_5p"),
            
            # 3. 努力 vs 结果 (Effort vs Result) - 威科夫理论的核心
            # 逻辑：如果 NetVol 很大 (努力大)，但 PriceChange 很小 (结果小) -> 吸收/滞涨
            # 计算：NetVol / PriceChange (取绝对值观察异常)
            (pl.col("taker_net_vol") / (pl.col("price_change").abs() + 1e-9))
            .alias("feat_absorption_ratio")
        ])