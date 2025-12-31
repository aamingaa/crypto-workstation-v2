import pandas as pd
import polars as pl
import numpy as np
from typing import Optional, List
import re

class LiquidationFactorEngine:
    def __init__(self, resample_freq='15m'):
        """
        优化版爆仓因子挖掘引擎
        架构: Polars (Tick处理/聚合) -> Pandas (特征挖掘/时序分析)
        """
        # Polars 的时间别名略有不同: '15T' -> '15m'
        # self.freq = resample_freq.replace('T', 'm') 
        self.freq = '15m'

    def process(self, raw_df_pd, 
                bucket_quantiles=[0.50, 0.90], 
                bucket_window_hours=[24],
                mining_windows=[24, 96, 672], 
                mining_quantiles=[0.90, 0.95, 0.99]):
        
        print(f"[*] 启动高性能引擎 (Polars Core) | 频率: {self.freq}")
        
        # --- 阶段一：Polars 高性能聚合 (Tick -> K-Line) ---
        # 1. 转换 Pandas -> Polars LazyFrame
        # 兼容：open_time 可能在列里，也可能在 DatetimeIndex 里（from_pandas 默认不带 index）
        raw_df_pd = raw_df_pd.copy()
        if "open_time" not in raw_df_pd.columns:
            if isinstance(raw_df_pd.index, pd.DatetimeIndex):
                # reset_index 后列名可能是：
                # - index 有名字：该名字
                # - index 无名字：默认 'index'
                tmp = raw_df_pd.reset_index()
                if "open_time" in tmp.columns:
                    raw_df_pd = tmp
                elif "index" in tmp.columns:
                    raw_df_pd = tmp.rename(columns={"index": "open_time"})
                else:
                    # 兜底：把 reset_index 生成的第一列当作时间列
                    raw_df_pd = tmp.rename(columns={tmp.columns[0]: "open_time"})
            else:
                raise ValueError("raw_df_pd 必须包含 'open_time' 列，或使用 DatetimeIndex 作为索引")

        # 显式转换时间列，防止类型不匹配
        raw_df_pd["open_time"] = pd.to_datetime(raw_df_pd["open_time"], errors="coerce")
        if raw_df_pd["open_time"].isna().any():
            raise ValueError("open_time 存在无法解析为时间的值，请先清洗/转换")
            
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

    def _polars_dynamic_bucketing_and_agg(self, lf, quantiles, lookback_hours : Optional[List[int]] = None):
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

        # B/C. 多窗口滚动平滑阈值 + ASOF join + 打标签
        th_base_names = [f"th_{int(q*100)}" for q in quantiles]
        lf_labeled = lf

        for w in lookback_hours:
            th_suffix = f"_roll_lb{w}"
            bucket_col = f"size_bucket_lb{w}"

            hourly_thresholds_w = (
                hourly_stats
                .with_columns([
                    pl.col(name).rolling_mean(window_size=w, min_samples=1)
                    .shift(1)
                    .alias(f"{name}{th_suffix}")
                    for name in th_base_names
                ])
                .select(["open_time"] + [f"th_{int(q*100)}{th_suffix}" for q in quantiles])
            )

            low_col = f"th_{int(quantiles[0]*100)}{th_suffix}"
            high_col = f"th_{int(quantiles[-1]*100)}{th_suffix}"

            lf_labeled = (
                lf_labeled
                .join_asof(hourly_thresholds_w, on="open_time", strategy="backward")
                .with_columns(
                    pl.when(pl.col("value") > pl.col(high_col)).then(pl.lit("large"))
                    .when(pl.col("value") <= pl.col(low_col)).then(pl.lit("small"))
                    .otherwise(pl.lit("med"))
                    .alias(bucket_col)
                )
            )
        
        # D. 聚合
        agg_exprs = []
        
        # 1. Sum & Count
        for w in lookback_hours:
            bucket_col = f"size_bucket_lb{w}"
            out_suffix = f"_lb{w}"

            for side in ['long', 'short']:
                for size in ['large', 'small', 'med']:
                    agg_exprs.append(
                        pl.col("value")
                        .filter((pl.col("liq_type") == side) & (pl.col(bucket_col) == size))
                        .sum()
                        .alias(f"sum_{side}_{size}{out_suffix}")
                    )
                    agg_exprs.append(
                        pl.col("value")
                        .filter((pl.col("liq_type") == side) & (pl.col(bucket_col) == size))
                        .count()
                        .alias(f"count_{side}_{size}{out_suffix}")
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
        
        cols = list(df.columns)

        # 2. 支持多 lookback 版本的派生（通过列名后缀 _lb{w} 识别）
        # 仅支持带 _lb{w} 后缀的版本（不再输出无后缀主窗口列）
        variants: set[str] = set()
        pattern = re.compile(r"^sum_long_(?:large|small|med)_lb(\d+)$")
        for c in cols:
            m = pattern.match(c)
            if m:
                variants.add(m.group(1))

        def _sum_cols(prefix: str, variant: str) -> List[str]:
            return [f"{prefix}_large_lb{variant}", f"{prefix}_small_lb{variant}", f"{prefix}_med_lb{variant}"]

        for v in sorted(variants, key=lambda x: int(x)):
            suffix = f"_lb{v}"

            long_sum_cols = [c for c in _sum_cols("sum_long", v) if c in df.columns]
            short_sum_cols = [c for c in _sum_cols("sum_short", v) if c in df.columns]

            if not long_sum_cols and not short_sum_cols:
                continue

            df[f"total_vol_long{suffix}"] = df[long_sum_cols].sum(axis=1) if long_sum_cols else 0.0
            df[f"total_vol_short{suffix}"] = df[short_sum_cols].sum(axis=1) if short_sum_cols else 0.0

            # 净爆仓压力 (Net Burn)
            df[f"diff_net_burn_vol{suffix}"] = df[f"total_vol_long{suffix}"] - df[f"total_vol_short{suffix}"]

            # 鲸鱼比例 (Whale Ratio)
            long_large = f"sum_long_large{suffix}"
            short_large = f"sum_short_large{suffix}"
            if long_large in df.columns:
                df[f"ratio_whale_long{suffix}"] = df[long_large] / (df[f"total_vol_long{suffix}"] + 1e-9)
            if short_large in df.columns:
                df[f"ratio_whale_short{suffix}"] = df[short_large] / (df[f"total_vol_short{suffix}"] + 1e-9)

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