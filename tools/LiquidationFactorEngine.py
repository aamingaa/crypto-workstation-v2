import pandas as pd
import polars as pl
import numpy as np

class LiquidationFactorEngineOptimized:
    def __init__(self, resample_freq='15m'):
        """
        优化版爆仓因子挖掘引擎
        架构: Polars (Tick处理/聚合) -> Pandas (特征挖掘/时序分析)
        """
        # Polars 的时间别名略有不同: '15T' -> '15m'
        self.freq = resample_freq.replace('T', 'm') 

    def process(self, raw_df_pd, 
                bucket_quantiles=[0.50, 0.90], 
                bucket_window_hours=24,
                mining_windows=[24, 96, 672], 
                mining_quantiles=[0.90, 0.95, 0.99]):
        
        print(f"[*] 启动高性能引擎 (Polars Core) | 频率: {self.freq}")
        
        # --- 阶段一：Polars 高性能聚合 (Tick -> K-Line) ---
        # 1. 转换 Pandas -> Polars LazyFrame
        # 显式转换时间列，防止类型不匹配
        if not np.issubdtype(raw_df_pd['open_time'].dtype, np.datetime64):
            raw_df_pd['open_time'] = pd.to_datetime(raw_df_pd['open_time'])
            
        lf = pl.from_pandas(raw_df_pd).lazy()
        
        # 2. 预计算 Value 和 基础映射
        # Tardis/Binance mapping: Side 'buy' = Short Liquidation (空头爆仓)
        lf = lf.with_columns([
            (pl.col("price") * pl.col("amount")).alias("value"),
            pl.col("side").replace({"buy": "short", "sell": "long"}).alias("liq_type")
        ]).sort("open_time") # 必须排序以支持 asof join

        # 3. 动态分桶 & 基础聚合 (核心优化点)
        df_agg = self._polars_dynamic_bucketing_and_agg(
            lf, bucket_quantiles, bucket_window_hours
        )
        
        # 将聚合后的 K线数据转回 Pandas 进行复杂的时序挖掘
        # (此时数据量已从千万级 Tick 降维到几万行 K线，Pandas 处理足够快且生态更好)
        df_kline = df_agg.to_pandas().set_index('open_time').sort_index()
        
        # --- 阶段二：Pandas 特征挖掘 (K-Line -> Factors) ---
        
        # 4. 衍生博弈特征
        print("[-] 生成博弈与比率特征...")
        df_derived = self._generate_derived_features(df_kline)
        
        # 5. 多维度时序挖掘 (Z-Score + Breakout)
        print(f"[-] 执行多维度挖掘 (Windows={mining_windows})...")
        df_final = self._cross_dimensional_mining(df_derived, mining_windows, mining_quantiles)
        
        print(f"[+] 处理完成. 输出因子数量: {df_final.shape[1]}")
        return df_final

    def _polars_dynamic_bucketing_and_agg(self, lf, quantiles, lookback_hours):
        """
        [修正版] 移除了 check_sorted 参数以适配 LazyFrame API
        """
        # A. 计算每小时的分布阈值
        hourly_stats = (
            lf.group_by_dynamic("open_time", every="1h")
            .agg([
                pl.col("value").quantile(q).alias(f"th_{int(q*100)}") 
                for q in quantiles
            ])
        )
        
        # B. 滚动平滑 & Shift (避免未来函数)
        rolling_cols = [pl.col(f"th_{int(q*100)}") for q in quantiles]
        
        hourly_thresholds = (
            hourly_stats
            .with_columns([
                c.rolling_mean(window_size=lookback_hours, min_periods=1)
                 .shift(1) 
                 .name.suffix("_roll")
                for c in rolling_cols
            ])
            .select(["open_time"] + [f"th_{int(q*100)}_roll" for q in quantiles])
        )

        # C. 极速合并 (ASOF Join) & 打标签
        low_col = f"th_{int(quantiles[0]*100)}_roll"
        high_col = f"th_{int(quantiles[-1]*100)}_roll"
        
        lf_labeled = (
            lf.join_asof(hourly_thresholds, on="open_time", strategy="backward")
            .with_columns(
                pl.when(pl.col("value") > pl.col(high_col)).then(pl.lit("large"))
                .when(pl.col("value") <= pl.col(low_col)).then(pl.lit("small"))
                .otherwise(pl.lit("med"))
                .alias("size_bucket")
            )
        )
        
        # D. 聚合
        agg_exprs = []
        
        # 1. Sum & Count
        for side in ['long', 'short']:
            for size in ['large', 'small', 'med']:
                agg_exprs.append(
                    pl.col("value")
                    .filter((pl.col("liq_type") == side) & (pl.col("size_bucket") == size))
                    .sum()
                    .alias(f"sum_{side}_{size}")
                )
                agg_exprs.append(
                    pl.col("value")
                    .filter((pl.col("liq_type") == side) & (pl.col("size_bucket") == size))
                    .count()
                    .alias(f"count_{side}_{size}")
                )
        
        # 2. Distribution Shape
        for side in ['long', 'short']:
            agg_exprs.append(
                pl.col("value")
                .filter(pl.col("liq_type") == side)
                .skew()
                .fill_nan(0)
                .alias(f"skew_{side}")
            )
        
        # [修正点] 移除了 check_sorted=False
        result = (
            lf_labeled
            .group_by_dynamic("open_time", every=self.freq) 
            .agg(agg_exprs)
            .collect() 
        )
        
        return result

    def _generate_derived_features(self, df):
        """
        Pandas 阶段：计算逻辑特征
        """
        df = df.copy()
        
        # 1. 基础填补
        df = df.fillna(0)
        
        # 2. 净爆仓压力 (Net Burn) - 核心博弈指标
        # 聚合所有 size 的总和
        cols = df.columns
        long_sum_cols = [c for c in cols if 'sum_long' in c]
        short_sum_cols = [c for c in cols if 'sum_short' in c]
        
        df['total_vol_long'] = df[long_sum_cols].sum(axis=1)
        df['total_vol_short'] = df[short_sum_cols].sum(axis=1)
        
        df['diff_net_burn_vol'] = df['total_vol_long'] - df['total_vol_short']
        
        # 3. 鲸鱼比例 (Whale Ratio)
        if 'sum_long_large' in df.columns:
            df['ratio_whale_long'] = df['sum_long_large'] / (df['total_vol_long'] + 1e-9)
        if 'sum_short_large' in df.columns:
            df['ratio_whale_short'] = df['sum_short_large'] / (df['total_vol_short'] + 1e-9)
            
        return df

    def _cross_dimensional_mining(self, df, windows, quantiles):
        """
        Pandas 阶段：时序挖掘
        增加了 Z-Score 标准化，以保留方向性
        """
        df_mining = df.copy()
        
        # 筛选需要挖掘的基础列
        target_candidates = [
            c for c in df.columns 
            if any(x in c for x in ['sum_', 'count_', 'diff_', 'skew_']) 
            and 'total' not in c # 避免重复
        ]
        
        for w in windows:
            # 预计算 Rolling 对象
            roller = df[target_candidates].rolling(window=w, min_periods=max(1, w//2))
            
            # --- 方法 A: Z-Score (标准化异常度) ---
            # 适用于 diff 类 (有正负) 和 skew 类
            # 逻辑: (当前值 - 历史均值) / 历史波动率
            # 相比分位数，这保留了符号信息：+3代表正向极值，-3代表负向极值
            mu = roller.mean().shift(1)
            sigma = roller.std().shift(1)
            
            z_scores = (df[target_candidates] - mu) / (sigma + 1e-9)
            z_scores.columns = [f"feat_z_{c}_w{w}" for c in target_candidates]
            
            # --- 方法 B: Breakout Ratio (单边突破) ---
            # 适用于 sum/count 类 (只有正数)，捕捉纯粹的量级爆发
            # 逻辑: 当前值 / 历史99分位
            ratio_feats = pd.DataFrame(index=df.index)
            # 只对非负特征做 Ratio 挖掘
            positive_targets = [c for c in target_candidates if 'diff' not in c and 'skew' not in c]
            
            for q in quantiles:
                thresh = df[positive_targets].rolling(w, min_periods=1).quantile(q).shift(1)
                ratio_part = df[positive_targets] / (thresh + 1e-9)
                ratio_part.columns = [f"feat_brk_{c}_w{w}_q{int(q*100)}" for c in positive_targets]
                ratio_feats = pd.concat([ratio_feats, ratio_part], axis=1)

            # 合并
            df_mining = pd.concat([df_mining, z_scores, ratio_feats], axis=1)

        return df_mining

# ==========================================
# 验证代码
# ==========================================
# if __name__ == "__main__":
#     # 生成更多数据以测试 Polars 优势
#     print("生成模拟数据 (50,000 Ticks)...")
#     dates = pd.date_range('2024-01-01', periods=50000000, freq='10s') 
#     raw_data = pd.DataFrame({
#         'open_time': dates,
#         'price': 60000 + np.random.randn(50000000).cumsum(),
#         'amount': np.random.exponential(scale=1.0, size=50000000) * np.random.choice([1, 10, 50], 50000000),
#         'side': np.random.choice(['buy', 'sell'], 50000000)
#     })
    
#     engine = LiquidationFactorEngineOptimized(resample_freq='15T')
    
#     import time
#     t0 = time.time()
#     df_factors = engine.process(raw_data)
#     print(f"耗时: {time.time() - t0:.2f} 秒")
    
#     print("\n[因子列预览]")
#     # 展示 Z-Score 和 Breakout 两种类型的因子
#     demo_cols = [c for c in df_factors.columns if 'feat_z_diff_net' in c or 'feat_brk_sum_long' in c]
#     print(df_factors[demo_cols].tail(3).T)