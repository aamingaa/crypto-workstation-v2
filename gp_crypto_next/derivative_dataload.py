import pandas as pd
import numpy as np

# ==========================================
# 0. 配置参数 (Configuration)
# ==========================================
CONFIG = {
    'timeframe': '15min',
    'rolling_window_short': 96,   # 短周期分位数窗口: 24小时 (15min * 4 * 24)
    'rolling_window_long': 672,   # 长周期分位数窗口: 7天 (用于捕捉长期位置)
    'liq_decay': 0.5              # (可选) 爆仓数据的半衰期权重，这里暂不使用，用Rank代替
}

# ==========================================
# 1. 模拟数据生成器 (Mock Data Generator)
# ==========================================
# 实际使用时，请用 pd.read_csv() 替换这部分
def generate_mock_data():
    print("正在生成模拟数据...")
    # 时间范围：最近 30 天
    dates = pd.date_range(start='2025-01-01', end='2025-02-01', freq='1s')
    
    # --- A. 模拟 K 线 (15min) ---
    dates_15m = pd.date_range(start='2025-01-01', end='2025-02-01', freq='15min')
    price_walk = np.cumprod(1 + np.random.normal(0, 0.002, len(dates_15m))) * 3000
    df_kline = pd.DataFrame({
        'open_time': dates_15m,
        'open': price_walk,
        'high': price_walk * 1.005,
        'low': price_walk * 0.995,
        'close': price_walk * (1 + np.random.normal(0, 0.001, len(dates_15m))),
        'volume': np.random.randint(1000, 100000, len(dates_15m))
    }).set_index('open_time')

    # --- B. 模拟高频爆仓流 (Liquidation Stream) ---
    # 随机生成 5000 条爆仓数据
    liq_indices = np.random.choice(dates, 5000)
    df_liq = pd.DataFrame({
        'open_time': liq_indices,
        'side': np.random.choice(['buy', 'sell'], 5000), # buy=空头爆, sell=多头爆
        'price': 3000 + np.random.normal(0, 50, 5000),
        'amount': np.abs(np.random.exponential(1.0, 5000)) # 数量符合指数分布
    }).sort_values('open_time')
    
    # --- C. 模拟资金费率/OI (Funding/OI) ---
    # 每分钟一条
    dates_1m = pd.date_range(start='2025-01-01', end='2025-02-01', freq='1min')
    df_funding = pd.DataFrame({
        'open_time': dates_1m,
        'funding_rate': np.random.normal(0.0001, 0.00005, len(dates_1m)),
        'open_interest': np.abs(np.cumsum(np.random.normal(0, 100, len(dates_1m))) + 10000),
        'mark_price': np.nan, # 暂用 close 模拟
        'index_price': np.nan 
    })
    # 简单的价格模拟
    price_interp = np.interp(dates_1m.astype(np.int64), dates_15m.astype(np.int64), df_kline['close'])
    df_funding['mark_price'] = price_interp
    df_funding['index_price'] = price_interp * 0.9995 # 现货略低
    
    return df_kline, df_liq, df_funding

# ==========================================
# 2. 核心清洗逻辑 (Data Cleaning Pipeline)
# ==========================================
def process_data(df_kline, df_liq, df_funding):
    print("开始执行清洗与聚合...")
    
    # --- A. 处理爆仓数据 ---
    df_liq['open_time'] = pd.to_datetime(df_liq['open_time'])
    df_liq.set_index('open_time', inplace=True)
    
    # [关键逻辑] 计算 USD 价值 & 区分方向
    # side == 'buy' -> 交易所买入 -> 用户做空被强平 -> Short Liq
    df_liq['usd_val'] = df_liq['price'] * df_liq['amount']
    df_liq['short_liq_vol'] = np.where(df_liq['side'] == 'buy', df_liq['usd_val'], 0)
    df_liq['long_liq_vol']  = np.where(df_liq['side'] == 'sell', df_liq['usd_val'], 0)
    
    # 聚合到 15min (求和)
    df_liq_15m = df_liq.resample(CONFIG['timeframe'], label='left', closed='left').agg({
        'short_liq_vol': 'sum',
        'long_liq_vol': 'sum',
        'usd_val': 'count' # 爆仓次数
    }).rename(columns={'usd_val': 'liq_count'})

    # --- B. 处理资金/OI数据 ---
    df_funding['open_time'] = pd.to_datetime(df_funding['open_time'])
    df_funding.set_index('open_time', inplace=True)
    
    # 聚合到 15min (状态量取 Last, 流量取 Mean)
    df_fund_15m = df_funding.resample(CONFIG['timeframe'], label='left', closed='left').agg({
        'open_interest': 'last',
        'funding_rate': 'mean',
        'mark_price': 'last',
        'index_price': 'last'
    })

    # --- C. 合并数据 ---
    # Left Join 以 K线时间为准
    df_final = pd.concat([df_kline, df_liq_15m, df_fund_15m], axis=1)
    
    # 基础填充
    # 爆仓没数据 = 0
    df_final['short_liq_vol'].fillna(0, inplace=True)
    df_final['long_liq_vol'].fillna(0, inplace=True)
    df_final['liq_count'].fillna(0, inplace=True)
    # 状态数据断流 = 延续上一个状态 (ffill)
    df_final.fillna(method='ffill', inplace=True)
    
    return df_final

# ==========================================
# 3. 因子工程 (Feature Engineering)
# ==========================================
def engineer_features(df):
    print("开始构建因子 (Feature Engineering)...")
    
    # 1. 基础价格特征 (Stationary)
    # 使用 Log Return 而不是价格绝对值
    df['ret'] = np.log(df['close'] / df['close'].shift(1))
    df['log_vol'] = np.log1p(df['volume']) # 压缩成交量量级
    
    # 2. 资金流特征
    # OI 变化率 (OI Momentum)
    df['oi_pct'] = df['open_interest'].pct_change()
    
    # 基差率 (Basis Ratio)
    # (合约 - 现货) / 现货
    df['basis'] = (df['mark_price'] - df['index_price']) / df['index_price']
    
    # 3. [核心] 滚动分位数 (Rolling Rank) - 解决非平稳性
    window = CONFIG['rolling_window_short'] # 96个bar = 24小时
    
    # 定义需要 Rank 的列
    cols_to_rank = {
        'short_liq_vol': 'rank_short_liq', # 空头爆仓强度
        'long_liq_vol':  'rank_long_liq',  # 多头爆仓强度
        'volume':        'rank_vol',       # 成交量强度
        'oi_pct':        'rank_oi_mom',    # 资金进场强度
        'funding_rate':  'rank_funding'    # 费率拥挤度
    }
    
    for raw_col, new_col in cols_to_rank.items():
        # rolling().rank(pct=True) 生成 0~1 之间的值
        df[new_col] = df[raw_col].rolling(window).rank(pct=True)
        # 处理初期 NaN: 如果窗口数据不足，默认给 0.5 (中性)
        df[new_col].fillna(0.5, inplace=True)
        
    # 4. [高阶] 爆仓失衡度 (Net Liquidation Imbalance)
    # 归一化到 [-1, 1] 区间
    # (空头爆 - 多头爆) / 总爆仓
    total_liq = df['short_liq_vol'] + df['long_liq_vol'] + 1e-6 # 防止除0
    df['liq_imbalance'] = (df['short_liq_vol'] - df['long_liq_vol']) / total_liq
    
    # 5. [目标变量] 用于 gplearn 训练的 Target
    # 预测未来 1 个 bar 的收益率 (shift(-1))
    df['target_next_ret'] = df['ret'].shift(-1)
    
    # 最后清洗：去除任何可能的 Inf 或 NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df

# ==========================================
# 4. 主程序运行
# ==========================================
if __name__ == "__main__":
    # 1. 获取数据
    raw_kline, raw_liq, raw_funding = generate_mock_data()
    
    # 2. 清洗
    df_clean = process_data(raw_kline, raw_liq, raw_funding)
    
    # 3. 构造因子
    df_features = engineer_features(df_clean)
    
    # 4. 预览输出
    print("\n" + "="*50)
    print("数据准备完成！可以输入给 gplearn 的特征列表：")
    print("="*50)
    
    feature_cols = [col for col in df_features.columns if 'rank_' in col or 'liq_imbalance' in col or 'basis' in col]
    print(f"特征数量: {len(feature_cols)}")
    print(f"特征列表: {feature_cols}")
    
    print("\n前 5 行数据预览:")
    print(df_features[feature_cols + ['close', 'target_next_ret']].head())
    
    # 5. (可选) 保存
    # df_features.to_csv("crypto_features_15m.csv")