import pandas as pd
import numpy as np
import talib as ta
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.figure(figsize=(10, 8))


# 计算双均线
def calculate_moving_averages(df, short_window, long_window):
    df['short_ma'] = ta.EMA(df['close'],short_window)
    df['long_ma'] = ta.EMA(df['close'],long_window)
    return df

def generate_trading_signals(data):
    data['signal'] = 0
    data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1
    data.loc[data['short_ma'] < data['long_ma'], 'signal'] = -1
    return data

# 波动率权重
def calculate_vol(df, target_var=0.01, window=500):
    # 计算收益率
    df['return'] = df['close'].pct_change()

    # 计算历史波动率
    df['rolling_std'] = df['return'].rolling(window=window).std()

    # 计算当前VaR
    mu = df['return'].rolling(window=window).mean()
    df['mean'] = mu

    cvars= []
    for i in range(len(df)):
        u = df['mean'].iloc[i]
        std = df['rolling_std'].iloc[i]
        current_var = norm.ppf(0.01, loc=u, scale=std)
        cvars.append(current_var)
    df['current_var'] = np.asarray(cvars)
    # 计算杠杆调整系数
    df['weight'] = 1*target_var / df['current_var'].abs()
    return df

def fit_gpd(drawdowns):
    """
    对回撤数据拟合广义帕累托分布（GPD）
    :param drawdowns: 回撤数据
    :return: GPD的形状和尺度参数
    """
    # 这里使用简单的方法估计GPD参数，实际应用中可使用更复杂的方法
    shape_param = np.mean(drawdowns)
    scale_param = np.std(drawdowns)
    return shape_param, scale_param


def calculate_cdar(drawdowns, shape_param, scale_param, confidence_level=0.95):
    """
    计算条件风险回撤（CDaR）
    :param drawdowns: 回撤数据
    :param shape_param: GPD形状参数
    :param scale_param: GPD尺度参数
    :param confidence_level: 置信水平
    :return: CDaR值
    """
    threshold = np.percentile(drawdowns, (1 - confidence_level) * 100)
    exceedances = drawdowns[drawdowns >= threshold]
    if shape_param != 0:
        cdar = threshold + (scale_param / shape_param) * \
               ((len(exceedances) / len(drawdowns) * (1 / (1 - confidence_level))) ** -shape_param - 1)
    else:
        cdar = threshold - scale_param * np.log(1 - confidence_level)
    return cdar


def cdar_pos(drawdowns,min_size=0.01,is_linear=False):
    drawdown = drawdowns[-1]

    if len(drawdowns)<10000 or is_linear:
        position_size = min_size + math.ceil(drawdown / 0.0006) * min_size if drawdown > 0 else min_size
        position_size = min(position_size, 3)
        return position_size
    shape_param, scale_param = fit_gpd(np.array(drawdowns))
    current_cdar = calculate_cdar(np.array(drawdowns), shape_param, scale_param)
    target_cdar = 0.1  # 设置目标CDaR
    position_size = abs(target_cdar / current_cdar)
    return position_size

def backtest_strategy(file_path):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['open_time'], unit='ms')
    data.set_index('date', inplace=True)
    data = calculate_moving_averages(data,short_window,long_window)
    data = generate_trading_signals(data)
    # data = calculate_vol(data)

    min_size = 0.01
    position_size = min_size
    initial_capital = 100000  # 初始资金
    capital = initial_capital
    total_return = 0
    drawdowns = []
    wealth_index = [initial_capital]
    last_signal = 0

    for i in range(len(data)):
        if i == 0:
            continue

        # 根据前一根 K 线的信号确定当前操作
        signal = data['signal'].iloc[i - 1]
        if pd.isna(position_size):
            position_size = 0
        # 计算当前 K 线的收益
        current_return = (data['close'].iloc[i] - data['close'].iloc[i - 1]) / data['close'].iloc[i - 1] * position_size * signal
        capital += capital * current_return
        total_return += current_return
        wealth_index.append(capital)

        # 计算回撤
        previous_peak = max(wealth_index)
        drawdown = (previous_peak - capital) / previous_peak
        drawdowns.append(drawdown)
        if signal!=last_signal:
            position_size = cdar_pos(drawdowns,is_linear=False)
        last_signal = signal

    # 计算最终的财富指数和累积收益
    wealth_index = np.array(wealth_index)
    cumulative_returns = np.asarray(wealth_index) / initial_capital
    print("最终财富指数:", wealth_index[-1])
    print("累积收益:", cumulative_returns)
    data['cumulative_returns'] = cumulative_returns
    data['cumulative_returns'].plot()
    plt.show()

short_window = 10
long_window = 30
# 运行回测
file_path = "/Users/aming/project/python/crypto-workstation-v2/tools/BTC_klines.csv"

backtest_strategy(file_path)