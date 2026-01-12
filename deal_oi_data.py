import pandas as pd
import numpy as np

def prepare_pure_microstructure_factors(df):
    """
    不包含任何 K 线价格因子的微观结构特征工程。
    专注于：OI、多空比、爆仓数据。
    """
    factors = pd.DataFrame(index=df.index)
    epsilon = 1e-9
    
    # =================================================================
    # 1. 深度多空博弈 (Sentiment & Divergence)
    # 逻辑：比较“资金态度”与“人数态度”的差异
    # =================================================================
    
    # [因子] 精英倾向指标 (Elite Bias)
    # 含义：如果 > 1，说明大户比散户更看多；如果 < 1，说明大户在做空而散户在做多
    # 这是捕捉“聪明钱”最直接的因子
    pos_ratio = df['topLongShortPositionRatio_longShortRatio_last']
    acc_ratio = df['topLongShortAccountRatio_longShortRatio_last']
    factors['smart_money_bias'] = pos_ratio / (acc_ratio + epsilon)
    
    # [因子] 情绪分歧度 (Sentiment Dispersion)
    # 含义：大户和散户的观点差异有多大（取绝对值）。差异越大，变盘概率越大。
    factors['sentiment_divergence_abs'] = np.abs(pos_ratio - acc_ratio)

    # [因子] 多空比日内波动率 (Sentiment Volatility)
    # 含义：这个小时内，多空比是不是上蹿下跳？代表情绪极其不稳定。
    factors['ls_ratio_volatility'] = (
        df['topLongShortPositionRatio_longShortRatio_max'] - 
        df['topLongShortPositionRatio_longShortRatio_min']
    ) / (pos_ratio + epsilon)

    # =================================================================
    # 2. 持仓量动力学 (OI Dynamics)
    # 逻辑：不看价格，只看资金是“进场打架”还是“离场观望”
    # =================================================================
    
    # [因子] OI 归一化震幅 (Normalized OI Noise)
    # 含义：OI 的 High-Low 差值代表资金的分歧激烈程度。
    # 相比单纯的 close-open，这个因子捕捉的是“盘中争夺”。
    factors['oi_intrabar_vol'] = (df['oi_high'] - df['oi_low']) / (df['oi_open'] + epsilon)
    
    # [因子] OI 趋势强度 (OI Strength)
    # 使用 log return 替代 pct_change，分布更正态，gplearn 更喜欢
    factors['oi_log_ret'] = np.log(df['oi_close'] / (df['oi_open'] + epsilon))

    # =================================================================
    # 3. 爆仓痛苦指数 (Liquidation Pain)
    # 逻辑：爆仓是市场阻力最小方向的燃料
    # =================================================================
    
    # [因子] 净爆仓方向 (Net Liquidation Imbalance)
    # -1 (全多头爆仓) ~ 1 (全空头爆仓)
    # 极其重要的反转因子。
    short_liq = df['short_turnover_sum']
    long_liq = df['long_turnover_sum']
    total_liq = short_liq + long_liq
    factors['liq_imbalance_norm'] = (short_liq - long_liq) / (total_liq + epsilon)
    
    # [因子] 爆仓相对强度 (Liquidation Intensity relative to OI)
    # 含义：这次爆仓对于当前的总持仓盘子来说，算不算“大动静”？
    # 解决了“牛市爆仓金额大，熊市爆仓金额小”的不可比问题。
    factors['liq_intensity_oi'] = total_liq / (df['oi_close'] + epsilon)

    # [因子] 爆仓价格分布偏度 (Liquidation Price Spread)
    # 利用你提供的 max/min/vwap 爆仓价
    # 逻辑：如果空头爆仓的 Max 价和 Min 价拉得非常大，说明这是一次长距离的“连环爆仓”（Cascade）。
    if 'short_liquidation_buy_price_max' in df.columns:
        factors['short_liq_spread'] = (
            df['short_liquidation_buy_price_max'] - df['short_liquidation_buy_price_min']
        ) / (df['short_liquidation_buy_price_first'] + epsilon)

    # =================================================================
    # 4. 统计特征增强 (Rolling Stats - Rank, Skew, Zscore)
    # 这一步对于 gplearn 至关重要，赋予数据“历史记忆”
    # =================================================================
    
    # 选择关键的 raw factors 进行滚动计算
    cols_to_roll = ['smart_money_bias', 'oi_log_ret', 'liq_imbalance_norm', 'liq_intensity_oi']
    
    # 窗口设置：例如 24 (1天), 168 (1周)
    windows = [24, 168] 
    
    for col in cols_to_roll:
        for w in windows:
            roller = factors[col].rolling(window=w, min_periods=w//2)
            
            # [核心] 滚动分位数 (Rolling Rank)
            # 含义：当前值处于过去 W 周期的一百分之几？(0~1)
            # 这是一个完美的归一化因子。
            factors[f'{col}_rank_{w}'] = roller.rank(pct=True)
            
            # [核心] 滚动 Z-Score
            # 含义：当前值偏离均值多少个标准差？
            factors[f'{col}_zscore_{w}'] = (factors[col] - roller.mean()) / (roller.std() + epsilon)
            
            # [核心] 滚动偏度 (Rolling Skew)
            # 含义：捕捉黑天鹅。例如 liq_intensity_oi 的偏度突然飙升，往往对应变盘点。
            factors[f'{col}_skew_{w}'] = roller.skew()

    # 清洗：去除计算产生的 NaN 和 Inf
    factors = factors.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return factors


if __name__ == "__main__":
    df = pd.DataFrame()
    prepare_pure_microstructure_factors(df)