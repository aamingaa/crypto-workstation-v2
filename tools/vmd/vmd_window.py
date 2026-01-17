import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
from sktime.libs.vmdpy import VMD

# 读取数据
df = pd.read_csv("/Users/aming/project/python/crypto-workstation-v2/tools/vmd/BTC_klines.csv")
prices = df['close'].values
df['return'] = df['close'].pct_change().fillna(0)
df['date'] = pd.to_datetime(df['open_time'], unit='ms')

# 定义滑动窗口大小和步长
window_size = 2000
step_size = 1

# 定义双均线策略的参数
short_window = 5
long_window = 20

# 初始化累计收益数组
cumulative_returns = np.zeros(len(prices))

pnl = []
dt = []
# 滑动窗口进行 VMD 分解和双均线策略回测
for start in range(window_size, len(prices), step_size):
    print(start)
    window_prices = prices[start-window_size:start]

    # 运行 VMD
    u, u_hat, omega = VMD(window_prices, alpha=2000, tau=0.0, K=4, DC=0, init=1, tol=1e-7)

    # 选择第一个模态函数作为信号序列
    signal = u[0]

    # 使用 talib 计算短期和长期移动平均
    short_mavg = talib.MA(signal, timeperiod=short_window)
    long_mavg = talib.MA(signal, timeperiod=long_window)

    # 生成交易信号
    signal = 1 if short_mavg[-1] > long_mavg[-1]  else -1


    strategy_returns = signal * df['return'].iloc[start]
    pnl.append(strategy_returns)
    dt.append(df['date'].iloc[start-1])

# 计算整体累计收益
pnl = np.asarray(pnl)
overall_cumulative_returns = (1 + pnl).cumprod()

# 绘制结果
plt.figure(figsize=(14, 7))
plt.plot(dt,overall_cumulative_returns, label='Strategy Cumulative Returns', alpha=0.7)
plt.legend()
plt.title('Comparison of Strategies')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.show()