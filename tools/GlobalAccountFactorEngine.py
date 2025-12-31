import polars as pl

class RetailSentimentEngine:
    def __init__(self, kline_df: pl.DataFrame, resample_freq='1h'):
        """
        :param kline_df: K线数据
        :param resample_freq: 建议使用 1h 或 4h，因为散户情绪变化较慢
        """
        self.freq = resample_freq
        self.kline_context = (
            kline_df.lazy()
            .select([pl.col("open_time"), pl.col("close")])
            .sort("open_time")
            .collect()
        )

    def process(self, raw_retail_df: pl.DataFrame, raw_whale_df: pl.DataFrame = None):
        """
        :param raw_retail_df: 你刚刚提供的 Global Account 数据
        :param raw_whale_df: 之前的 Top Trader 数据 (用于计算剪刀差)
        """
        print(f"[*] 启动散户情绪引擎 | 频率: {self.freq}")
        
        # 1. 清洗散户数据
        lf_retail = self._clean_data(raw_retail_df, "retail")
        
        # 2. 基础特征 (情绪极端度)
        lf_retail = self._build_sentiment_features(lf_retail)
        
        # 3. 如果提供了大户数据，计算“剪刀差” (核心 Alpha)
        if raw_whale_df is not None:
            print("[-] 检测到大户数据，正在计算【精英-散户背离】因子...")
            lf_whale = self._clean_data(raw_whale_df, "whale")
            
            # 对齐数据
            lf_merged = lf_retail.join(lf_whale, on="open_time", how="inner")
            
            # 计算剪刀差
            lf_final = self._build_divergence_features(lf_merged)
        else:
            lf_final = lf_retail

        return lf_final.collect()

    def _clean_data(self, df: pl.DataFrame, prefix: str):
        """
        通用清洗：去重、时间解析、重采样
        """
        # 识别时间字段名 (timestamp 或 open_time)
        time_col = "timestamp" if "timestamp" in df.columns else "open_time"
        
        q = (
            df.lazy()
            .unique(subset=[time_col], keep="last")
        )
        
        # 处理时间戳 (你的数据是 Unix ms)
        if time_col == "timestamp":
            q = q.with_columns(
                pl.col("timestamp").cast(pl.Int64).cast(pl.Datetime("ms")).alias("open_time")
            )
        else:
            q = q.with_columns(
                pl.col("open_time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
            )
            
        return (
            q.sort("open_time")
            .group_by_dynamic("open_time", every=self.freq)
            .agg([
                # 计算净头寸: Long% - Short%
                (pl.col("longAccount").last().cast(pl.Float64) - 
                 pl.col("shortAccount").last().cast(pl.Float64)).alias(f"{prefix}_net_pos")
            ])
        )

    def _build_sentiment_features(self, lf: pl.LazyFrame):
        """
        散户的反向指标逻辑
        """
        return lf.with_columns([
            # 1. 散户极端情绪 (Z-Score)
            # 如果 Z > 2，说明散户过度看多 -> 危险
            ((pl.col("retail_net_pos") - pl.col("retail_net_pos").rolling_mean(168)) / 
             (pl.col("retail_net_pos").rolling_std(168) + 1e-9))
            .alias("feat_retail_sentiment_zscore_7d")
        ])

    def _build_divergence_features(self, lf: pl.LazyFrame):
        """
        【皇冠上的明珠】计算 Smart Money vs Retail 的剪刀差
        """
        return lf.with_columns([
            # 1. 剪刀差 (Smart - Dumb Spread)
            # 正值 = 大户比散户更看多 (做多信号)
            # 负值 = 大户比散户更看空 (做空信号)
            (pl.col("whale_net_pos") - pl.col("retail_net_pos")).alias("feat_smart_dumb_spread"),
            
            # 2. 剪刀差的变化率 (Spread Momentum)
            (pl.col("whale_net_pos") - pl.col("retail_net_pos")).diff().alias("feat_spread_change")
        ])