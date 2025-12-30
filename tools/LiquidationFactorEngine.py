import pandas as pd
import numpy as np

class LiquidationFactorEngine:
    def __init__(self, resample_freq='15T'):
        """
        初始化爆仓因子挖掘引擎
        :param resample_freq: 输出K线的时间频率，推荐 '15T'(15分钟) 或 '1H'(1小时)
        """
        self.freq = resample_freq

    def process(self, raw_df, 
                bucket_quantiles=[0.50, 0.90], 
                bucket_window_hours=24,
                mining_windows=[24, 96, 672], 
                mining_quantiles=[0.90, 0.95, 0.99]):
        """
        全流程处理主函数
        
        :param raw_df: 原始数据 (需包含 price, amount, side, open_time)
        :param bucket_quantiles: 定义大中小单的动态分位阈值 (如 [0.5, 0.9])
        :param bucket_window_hours: 计算动态阈值的回看窗口 (单位: 小时)
        :param mining_windows: 挖掘异常特征的滚动窗口 (单位: resample_freq 的个数)
        :param mining_quantiles: 挖掘异常特征的历史分位阈值
        """
        print(f"[*] 启动引擎 | 频率: {self.freq} | 动态分桶回看: {bucket_window_hours}H")
        
        # 1. 微观分桶 (Micro-Structure)
        # 给每一笔 Tick 打上标签: size='large/small', type='long/short'
        df_labeled = self._dynamic_bucketing(raw_df, bucket_quantiles, bucket_window_hours)
        
        # 2. 基础聚合 (Base Stats) -> Sum, Count
        # 输出: sum_long_large, count_short_small
        print("[-] 正在聚合基础统计量 (Sum, Count)...")
        df_agg = self._aggregate_basic_stats(df_labeled)
        
        # 3. 形态特征 (Distribution Shape) -> Skew, Kurt
        # 输出: skew_long, kurt_short (不分大小单，只分方向以保证样本量)
        print("[-] 正在计算分布形态 (Skew, Kurt)...")
        df_shape = self._calc_distribution_shape(df_labeled)
        
        # 合并基础表
        df_combined = pd.concat([df_agg, df_shape], axis=1).fillna(0)
        
        # 4. 衍生博弈特征 (Ratios & Diffs)
        # 输出: whale_dominance, net_burn_vol
        print("[-] 生成博弈与比率特征...")
        df_derived = self._generate_derived_features(df_combined)
        
        # 5. 多维度时序挖掘 (Cross-Dimensional Mining)
        # 输出: feat_sum_long_large_w96_q99 (历史突破因子)
        print(f"[-] 执行多维度挖掘 (Windows={mining_windows})...")
        df_final = self._cross_dimensional_mining(df_derived, mining_windows, mining_quantiles)
        
        print(f"[+] 处理完成. 输出因子数量: {df_final.shape[1]}")
        return df_final

    def _dynamic_bucketing(self, df, quantiles, lookback_hours):
        """
        阶段一：动态分桶
        基于过去 N 小时的订单分布，自适应定义大中小单
        """
        df = df.copy()
        # 确保时间格式
        # if not np.issubdtype(df['open_time'].dtype, np.datetime64):
        #     df['open_time'] = pd.to_datetime(df['open_time'])
            
        df['value'] = df['price'] * df['amount']
        # df.set_index('open_time', inplace=True)
        
        # A. 计算每小时的分布阈值
        # 这里的 1H 是为了计算阈值的粒度，不影响最终输出频率
        hourly_quantiles = df['value'].resample('1H').quantile(quantiles).unstack()
        
        # B. 滚动平滑阈值 (Shift防止未来函数)
        # 含义: "当前的大单标准，参考过去24小时的平均水平"
        rolling_thresh = hourly_quantiles.rolling(window=lookback_hours, min_periods=1).mean().shift(1)
        rolling_thresh.fillna(method='bfill', inplace=True)
        
        # C. 将阈值映射回 Tick 数据
        df['h_key'] = df.index.floor('H')
        rolling_thresh.columns = [f'th_{int(q*100)}' for q in rolling_thresh.columns]
        df_merged = df.merge(rolling_thresh, left_on='h_key', right_index=True, how='left')
        
        # D. 向量化打标签
        # 逻辑: <= Low_Q (Small), > High_Q (Large), else (Med)
        low_col = f'th_{int(quantiles[0]*100)}'
        high_col = f'th_{int(quantiles[-1]*100)}'
        
        conditions = [
            df_merged['value'] > df_merged[high_col],
            df_merged['value'] <= df_merged[low_col]
        ]
        choices = ['large', 'small']
        df_merged['size_bucket'] = np.select(conditions, choices, default='med')
        
        # E. 映射方向 (Tardis: side='buy'是空头爆仓, 'sell'是多头爆仓)
        df_merged['liq_type'] = df_merged['side'].map({'buy': 'short', 'sell': 'long'})
        
        return df_merged

    def _aggregate_basic_stats(self, df):
        """
        阶段二：基础聚合 (Sum, Count)
        按 Time + Direction + Size 分组
        """
        grouped = df.groupby([
            pd.Grouper(freq=self.freq),
            'liq_type',
            'size_bucket'
        ])['value'].agg(['sum', 'count']).unstack(level=[1, 2])
        
        # 展平列名 e.g., sum_long_large
        grouped.columns = [f"{stat}_{side}_{size}" for stat, side, size in grouped.columns]
        return grouped

    def _calc_distribution_shape(self, df):
        """
        阶段三：形态特征 (Skew, Kurt)
        按 Time + Direction 分组 (合并大小单以确保统计稳定性)
        """
        # 自定义聚合函数
        def safe_skew(x): return x.skew() if len(x) >= 5 else 0
        def safe_kurt(x): return x.kurt() if len(x) >= 5 else 0
        
        grouped = df.groupby([
            pd.Grouper(freq=self.freq),
            'liq_type'
        ])['value'].agg([
            ('skew', safe_skew),
            ('kurt', safe_kurt)
        ]).unstack(level=1)
        
        # 展平列名 e.g., skew_long, kurt_short
        grouped.columns = [f"{stat}_{side}" for stat, side in grouped.columns]
        return grouped

    def _generate_derived_features(self, df):
        """
        阶段四：衍生博弈特征 (Ratios & Diffs)
        """
        df = df.copy()
        
        # 辅助: 获取所有相关的列
        cols = df.columns
        
        # 1. 鲸鱼集中度 (Whale Dominance)
        # sum_long_large / sum_long_total
        for side in ['long', 'short']:
            large_col = f'sum_{side}_large'
            total_cols = [c for c in cols if f'sum_{side}' in c]
            if large_col in df.columns:
                total_val = df[total_cols].sum(axis=1)
                df[f'ratio_whale_{side}'] = df[large_col] / (total_val + 1e-9)

        # 2. 散户恐慌度 (Retail Panic Ratio)
        # count_side_small / count_side_total
        for side in ['long', 'short']:
            small_col = f'count_{side}_small'
            count_cols = [c for c in cols if f'count_{side}' in c]
            if small_col in df.columns:
                total_cnt = df[count_cols].sum(axis=1)
                df[f'ratio_retail_{side}'] = df[small_col] / (total_cnt + 1e-9)

        # 3. 净爆仓压力 (Net Burn Pressure)
        # 多头总金额 - 空头总金额
        sum_long_all = df[[c for c in cols if 'sum_long' in c]].sum(axis=1)
        sum_short_all = df[[c for c in cols if 'sum_short' in c]].sum(axis=1)
        df['diff_net_burn_vol'] = sum_long_all - sum_short_all
        
        # 4. 净鲸鱼压力 (Net Whale Pressure)
        # 多头大单 - 空头大单
        if 'sum_long_large' in df.columns and 'sum_short_large' in df.columns:
            df['diff_net_whale_vol'] = df['sum_long_large'] - df['sum_short_large']

        # 5. 形态差异 (Shape Divergence)
        # 多头是不是比空头更像是"定点爆破"? (Skew差值)
        if 'skew_long' in df.columns and 'skew_short' in df.columns:
            df['diff_skew'] = df['skew_long'] - df['skew_short']

        return df

    def _cross_dimensional_mining(self, df, windows, quantiles):
        """
        阶段五：多维度交叉挖掘
        计算: Current_Value / Rolling_Quantile(History)
        """
        df_mining = df.copy()
        
        # 只对具有物理意义的量级特征做挖掘 (金额, 笔数, 差值)
        # 比率(Ratio)和形态(Skew)通常是平稳的，可以直接用，不需要做分位数比率
        target_candidates = [c for c in df.columns if any(x in c for x in ['sum_', 'count_', 'diff_'])]
        
        for w in windows:
            # 预计算滚动窗口对象
            roller = df[target_candidates].rolling(window=w, min_periods=max(1, w//2))
            
            for q in quantiles:
                # 1. 计算历史阈值 (Shifted 1 step)
                thresh = roller.quantile(q).shift(1)
                
                # 2. 计算当前值相对于历史阈值的倍数
                # 特征名: feat_{原名}_w{窗口}_q{分位}
                # e.g. feat_sum_long_large_w96_q99
                suffix = f"_w{w}_q{int(q*100)}"
                
                # 向量化计算
                # 添加 abs() 处理 diff 类的负数情况，或者保持符号看具体的逻辑
                # 这里我们假设关注"突破程度"，保留原始方向的比率可能不稳定，
                # 对于 sum/count 类 (正数)，直接除; 对于 diff 类 (有正负)，建议先取绝对值再挖掘其"极端程度"
                
                for col in target_candidates:
                    feat_name = f"feat_{col}{suffix}"
                    
                    if 'diff_' in col:
                        # 对于差值，我们关心的是"偏离度的绝对值"是否异常
                        # 或者计算 Z-Score 更好，这里为了统一用分位数逻辑:
                        # 检查 abs(current) / quantile(abs(history))
                        abs_series = df[col].abs()
                        abs_thresh = abs_series.rolling(window=w, min_periods=1).quantile(q).shift(1)
                        df_mining[feat_name] = abs_series / (abs_thresh + 1e-9)
                    else:
                        # 对于 sum/count，直接计算比率
                        df_mining[feat_name] = df[col] / (thresh[col] + 1e-9)
                        
        return df_mining

# ==========================================
# 模拟数据与运行示例
# ==========================================
if __name__ == "__main__":
    # 1. 生成模拟 Tardis 数据 (10000笔 Tick)
    print("生成模拟数据...")
    dates = pd.date_range('2024-01-01', periods=20000, freq='30S') # 覆盖几天
    raw_data = pd.DataFrame({
        'open_time': dates,
        # 价格随机游走
        'price': 50000 + np.random.randn(20000).cumsum(),
        # 金额符合指数分布 (大部分是小单, 偶尔有大单)
        'amount': np.random.exponential(scale=0.5, size=20000) * np.random.choice([1, 10, 100], 20000, p=[0.8, 0.15, 0.05]),
        # 方向随机
        'side': np.random.choice(['buy', 'sell'], 20000)
    })
    
    # 2. 初始化引擎 (按15分钟聚合)
    engine = LiquidationFactorEngine(resample_freq='15T')
    
    # 3. 运行处理
    # bucket_window_hours=24: 用过去24小时分布定义大单
    # mining_windows=[96, 672]: 挖掘过去 24小时(96*15m) 和 7天(672*15m) 的异常
    df_factors = engine.process(
        raw_data,
        bucket_quantiles=[0.50, 0.90], 
        bucket_window_hours=24,
        mining_windows=[96, 672], 
        mining_quantiles=[0.95, 0.99]
    )
    
    # 4. 查看结果
    print("\n[预览结果]")
    # 挑选几个核心列展示
    preview_cols = [
        'sum_long_large',          # 基础: 多头大单总额
        'skew_long',               # 形态: 多头爆仓偏度
        'diff_net_burn_vol',       # 博弈: 净爆仓压力
        'feat_sum_long_large_w96_q99' # 挖掘: 24H历史极值突破因子
    ]
    # 仅展示存在的列
    cols_to_show = [c for c in preview_cols if c in df_factors.columns]
    print(df_factors[cols_to_show].tail())