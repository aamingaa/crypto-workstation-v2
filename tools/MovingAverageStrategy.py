import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import math
import os

# 设置中文显示
plt.rcParams["font.family"] = ["Heiti TC", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


file_path = f'/Volumes/Ext-Disk/data/futures/um/daily/klines'
save_path = f'/Users/aming/project/python/crypto-workstation-v2/output/evaluate_metric'

def generate_date_range(start_date, end_date):    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    return date_list

def load_daily_data(file_dir:str = None, start_date:str = None, end_date:str = None, interval:str = None, crypto:str = "BNBUSDT") -> pd.DataFrame:
    date_list = generate_date_range(start_date, end_date)
    crypto_date_data = []
    if file_dir:
        crypto_date_data.append(pd.read_csv(file_dir))
    else:
        for date in date_list:
            suffix = "2025-01-01_2025-07-01"
            crypto_date_data.append(pd.read_csv(f"{file_path}/{crypto}/{interval}/{suffix}/{crypto}-{interval}-{date}.zip"))
    
    z = pd.concat(crypto_date_data, axis=0, ignore_index=True)
    # 处理时间戳为 DatetimeIndex，便于后续按日/月/年分组和年化计算
    z['open_time'] = pd.to_datetime(z['open_time'], unit='ms')
    z['close_time'] = pd.to_datetime(z['close_time'], unit='ms')

    z = z.sort_values(by='close_time', ascending=True) # 注意这一步是非常必要的，要以timestamp作为排序基准
    z = z.drop_duplicates('close_time').reset_index(drop=True) # 注意这一步非常重要，以timestamp为基准进行去重处理
    # z = z.set_index('close_time')
    z['interval'] = interval  # 保存interval信息，供后续使用
    return z


class MAStrategyAnalyzer:
    def __init__(self, data, short_window=5, long_window=20, crypto : str = None, commission_rate=0.0001):
        """
        初始化均线策略分析器
        :param data: 包含OHLC数据的DataFrame，需包含'open_time', 'open', 'high', 'low', 'close'列
        :param short_window: 短期均线周期
        :param long_window: 长期均线周期
        :param commission_rate: 手续费率，默认0.0001（万一）
        """
        self.data = data.copy()
        # 确保时间列是datetime类型
        self.data['open_time'] = pd.to_datetime(self.data['open_time'])
        self.data['close_time'] = pd.to_datetime(self.data['close_time'])
        
        self.short_window = short_window
        self.long_window = long_window
        self.commission_rate = commission_rate  # 手续费率（双向收取）
        
        # 计算均线
        self.data['short_ma'] = self.data['close'].rolling(window=short_window).mean()
        self.data['long_ma'] = self.data['close'].rolling(window=long_window).mean()
        
        # 初始化交易相关列
        self.data['signal'] = 0  # 1表示买入，-1表示卖出
        self.data['position'] = 0  # 1表示持有多单，0表示空仓
        self.data['strategy_return'] = 0.0  # 策略收益
        self.data['cumulative_return'] = 1.0  # 累计收益
        self.data['pnl'] = 0.0  # 每根K线的PNL（利润/亏损）
        self.data['cumulative_pnl'] = 0.0  # 累计PNL
        
        self.trades = []  # 记录所有交易
        self.crypto = crypto
        self.performance_metrics = {}  # 记录绩效指标
        self.total_pnl = 0.0  # 总PNL
        
    def generate_signals(self):
        """生成交易信号：短期均线上穿长期均线买入，下穿卖出，采用下一根K线开盘执行"""
        # 初始化信号列为0
        self.data['signal'] = 0
        
        # 检测金叉（Golden Cross）：短期均线从下方上穿长期均线，买入信号
        golden_cross = (self.data['short_ma'] > self.data['long_ma']) & \
                       (self.data['short_ma'].shift(1) <= self.data['long_ma'].shift(1))
        self.data.loc[golden_cross, 'signal'] = 1
        
        # 检测死叉（Death Cross）：短期均线从上方下穿长期均线，卖出信号
        death_cross = (self.data['short_ma'] < self.data['long_ma']) & \
                      (self.data['short_ma'].shift(1) >= self.data['long_ma'].shift(1))
        self.data.loc[death_cross, 'signal'] = -1
        
        # 采用“下一根K线开盘执行”：将信号右移一根作为执行信号
        exec_signal = self.data['signal'].shift(1).fillna(0)
        
        # 根据执行信号更新持仓：1 -> 持多，-1 -> 平仓（仅做多策略：-1 表示从1变为0）
        position = [0]
        self.trades = []
        self.total_pnl = 0.0
        self.data['pnl'] = 0.0
        
        for i in range(1, len(self.data)):
            prev_pos = position[-1]
            open_price = self.data['open'].iloc[i]
            open_time = self.data['open_time'].iloc[i]
            current_pnl = 0.0
            
            # 执行买入：上一根收盘出现买入信号，本根开盘以开盘价成交
            if exec_signal.iloc[i] == 1 and prev_pos == 0:
                position.append(1)
                buy_commission = open_price * self.commission_rate
                self.trades.append({
                    'type': 'buy',
                    'time': open_time,
                    'price': open_price,
                    'commission': buy_commission,
                    'index': i
                })
                current_pnl -= buy_commission
            # 执行卖出（平仓）：上一根收盘出现卖出信号，本根开盘以开盘价成交
            elif exec_signal.iloc[i] == -1 and prev_pos == 1:
                position.append(0)
                sell_commission = open_price * self.commission_rate
                if self.trades and self.trades[-1]['type'] == 'buy':
                    buy_trade = self.trades[-1]
                    gross_pnl = open_price - buy_trade['price']
                    net_pnl = gross_pnl - (sell_commission)
                    self.total_pnl += net_pnl
                    current_pnl = net_pnl  # 仅在平仓时记入实现盈亏
                    self.trades.append({
                        'type': 'sell',
                        'time': open_time,
                        'price': open_price,
                        'commission': sell_commission,
                        'gross_pnl': gross_pnl,
                        'net_pnl': net_pnl,
                        'holding_period': i - buy_trade['index'],
                        'index': i
                    })
                else:
                    self.trades.append({
                        'type': 'sell',
                        'time': open_time,
                        'price': open_price,
                        'commission': sell_commission,
                        'gross_pnl': 0.0,
                        'net_pnl': -sell_commission,
                        'holding_period': 0,
                        'index': i
                    })
                    self.total_pnl -= sell_commission
            else:
                position.append(prev_pos)
                # 非平仓时不记录浮动盈亏，避免重复统计
                current_pnl = 0.0
            
            self.data.at[i, 'pnl'] = current_pnl
        
        # 写回持仓序列
        self.data['position'] = position
        
        # 仅累积实现盈亏
        self.data['cumulative_pnl'] = self.data['pnl'].cumsum()
        
        # 采用开盘到开盘的收益率，并按上一根持仓产生策略收益
        self.data['market_return'] = self.data['open'].pct_change()
        self.data['strategy_return'] = self.data['market_return'] * self.data['position'].shift(1).fillna(0)
        self.data['cumulative_market'] = (1 + self.data['market_return'].fillna(0)).cumprod()
        self.data['cumulative_strategy'] = (1 + self.data['strategy_return'].fillna(0)).cumprod()
    
    def calculate_performance(self):
        """计算策略绩效指标（包含PNL相关指标）"""
        # 筛选出所有完整交易（买入后卖出）
        sell_trades = [t for t in self.trades if t['type'] == 'sell']
        profitable_trades = [t for t in sell_trades if t['net_pnl'] > 0]
        losing_trades = [t for t in sell_trades if t['net_pnl'] <= 0]
        
        # 胜率（基于净PNL）
        total_sells = len(sell_trades)
        win_rate = len(profitable_trades) / total_sells if total_sells > 0 else 0
        
        # PNL相关指标
        total_pnl = self.total_pnl
        # avg_pnl_per_trade = total_pnl / total_sells if total_sells > 0 else 0
        # avg_profit_pnl = np.mean([t['net_pnl'] for t in profitable_trades]) if profitable_trades else 0
        # avg_loss_pnl = np.mean([t['net_pnl'] for t in losing_trades]) if losing_trades else 0
        
        # 累计收益率
        total_return = self.data['cumulative_strategy'].iloc[-1] - 1 if len(self.data) > 0 else 0
        
        # 夏普比率（基于15分钟bar：每日96根，按365天年化）
        returns = self.data['strategy_return'].dropna()
        if len(returns) > 0 and returns.std() > 0:
            annualization = 96.0 * 365.0
            sharpe_ratio = math.sqrt(annualization) * (returns.mean() / returns.std())
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        cumulative = self.data['cumulative_strategy'].dropna()
        if len(cumulative) > 0:
            peak = cumulative.expanding(min_periods=1).max()
            drawdown = (cumulative / peak) - 1
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0
        
        # 最大PNL和最小PNL
        max_pnl = max([t['net_pnl'] for t in sell_trades]) if sell_trades else 0
        min_pnl = min([t['net_pnl'] for t in sell_trades]) if sell_trades else 0
        
        self.performance_metrics = {
            '总PNL': total_pnl,
            # '每笔交易平均PNL': avg_pnl_per_trade,
            # '平均盈利PNL': avg_profit_pnl,
            # '平均亏损PNL': avg_loss_pnl,
            '最大盈利PNL': max_pnl,
            '最大亏损PNL': min_pnl,
            '累计收益率': total_return,
            '夏普比率': sharpe_ratio,
            '最大回撤': max_drawdown,
            '胜率': win_rate,
            '总交易次数': total_sells,
            '总手续费支出': sum(t['commission'] for t in self.trades),
            'symbol': self.crypto
        }
        
        return self.performance_metrics
    
    def plot_results(self, save_path=None):
        """绘制策略结果，新增PNL曲线"""
        # 创建3个子图：价格+信号、累计收益、累计PNL
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14), sharex=True)
        
        plt.ion()

        # 1. 价格走势与交易信号
        ax1.plot(self.data['close_time'], self.data['close'], label='收盘价', linewidth=1.5)
        ax1.plot(self.data['close_time'], self.data['short_ma'], label=f'{self.short_window}周期均线', linewidth=1.2)
        ax1.plot(self.data['close_time'], self.data['long_ma'], label=f'{self.long_window}周期均线', linewidth=1.2)
        
        # 标记买入卖出点
        buy_signals = [t for t in self.trades if t['type'] == 'buy']
        sell_signals = [t for t in self.trades if t['type'] == 'sell']
        
        if buy_signals:
            buy_times = [t['time'] for t in buy_signals]
            buy_prices = [t['price'] for t in buy_signals]
            ax1.scatter(buy_times, buy_prices, marker='^', color='g', label='买入', s=100)
        
        if sell_signals:
            sell_times = [t['time'] for t in sell_signals]
            sell_prices = [t['price'] for t in sell_signals]
            ax1.scatter(sell_times, sell_prices, marker='v', color='r', label='卖出', s=100)
        
        ax1.set_title('价格走势与交易信号')
        ax1.set_ylabel('价格')
        ax1.legend()
        ax1.grid(True)
        
        # 2. 累计收益曲线
        ax2.plot(self.data['close_time'], self.data['cumulative_strategy'], label='策略累计收益', linewidth=1.5)
        ax2.plot(self.data['close_time'], self.data['cumulative_market'], label='市场累计收益', linewidth=1.5)
        ax2.set_title('策略与市场累计收益对比')
        ax2.set_ylabel('累计收益倍数')
        ax2.legend()
        ax2.grid(True)
        
        # plt.tight_layout()
        # # 3. 累计PNL曲线（新增）
        # ax3.plot(self.data['close_time'], self.data['cumulative_pnl'], label='累计PNL', linewidth=1.5, color='purple')
        # ax3.axhline(y=0, color='r', linestyle='--', alpha=0.3)  # 盈亏平衡线
        # ax3.set_title('累计PNL走势')
        # ax3.set_xlabel('时间 (close_time)')
        # ax3.set_ylabel('PNL')
        # ax3.legend()
        # ax3.grid(True)
        
        # 设置时间轴格式
        fig.autofmt_xdate()
        # ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        # plt.show()
    
    def run_strategy(self, save_plot_path=None):
        """运行完整策略并输出结果"""
        self.generate_signals()
        metrics = self.calculate_performance()
        
        print("策略参数:")
        print(f"短期均线周期: {self.short_window} (15分钟K线)")
        print(f"长期均线周期: {self.long_window} (15分钟K线)")
        print(f"手续费率: {self.commission_rate:.4%} (双向收取)")
        
        # print("\nPNL相关指标:")
        # pnl_metrics = ['总PNL', '每笔交易平均PNL', '平均盈利PNL', '平均亏损PNL', '最大盈利PNL', '最大亏损PNL']
        # for key in pnl_metrics:
        #     print(f"{key}: {metrics[key]:.4f}")
        
        if metrics is not None:
            print("\n其他绩效指标:")
            other_metrics = ['累计收益率', '夏普比率', '最大回撤', '胜率', '总交易次数', '总手续费支出']
            for key in other_metrics:
                if key in ['累计收益率', '胜率']:
                    print(f"{key}: {metrics[key]:.2%}")
                else:
                    print(f"{key}: {metrics[key]:.4f}")
        else:
            print("\n没有交易记录，可视化净值但不打印绩效指标。")
        
        self.plot_results(save_path=save_plot_path)

        return metrics

# 示例数据生成和使用
def generate_sample_data(n_periods=1000):
    """生成示例15分钟K线数据用于测试"""
    start_time = datetime(2023, 1, 1, 9, 30)
    times = pd.date_range(start=start_time, periods=n_periods, freq='15T')
    
    # 生成带趋势的价格数据
    base_price = 100.0
    trend = np.linspace(0, 50, n_periods)
    noise = np.random.normal(0, 1, n_periods).cumsum()
    
    close_prices = base_price + trend + noise
    open_prices = close_prices[:-1]
    open_prices = np.append(open_prices, close_prices[-1] * (1 + np.random.normal(0, 0.001)))
    
    high_prices = np.maximum(open_prices, close_prices) + np.random.uniform(0, 1, n_periods)
    low_prices = np.minimum(open_prices, close_prices) - np.random.uniform(0, 1, n_periods)
    
    return pd.DataFrame({
        'open_time': times,
        'close_time': times + pd.Timedelta(minutes=15),
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    })

if __name__ == "__main__":
    # 生成示例数据
    print("生成示例15分钟K线数据...")
    start_date = "2025-06-01"
    end_date = "2025-06-30"

    crypto_metric={}
    # crypto_list = ["ZECUSDT","XTZUSDT","BNBUSDT","ATOMUSDT","ONTUSDT","IOTAUSDT","BATUSDT","VETUSDT","NEOUSDT","QTUMUSDT","IOSTUSDT","THETAUSDT","ALGOUSDT","ZILUSDT","KNCUSDT","ZRXUSDT","COMPUSDT","DOGEUSDT","SXPUSDT","KAVAUSDT","BANDUSDT","RLCUSDT","SNXUSDT","DOTUSDT","YFIUSDT","CRVUSDT","TRBUSDT","RUNEUSDT","SUSHIUSDT","EGLDUSDT","SOLUSDT"]
    # crypto_list = ["BTCUSDT"]
    # crypto_list = ["ROSEUSDT","DUSKUSDT","FLOWUSDT","IMXUSDT"]
    crypto_list = ["ETHUSDT"]
    # crypto_list = ["BTCUSDT","ETHUSDT","BCHUSDT","XRPUSDT","LTCUSDT","TRXUSDT","ETCUSDT","LINKUSDT","XLMUSDT","ADAUSDT","XMRUSDT","DASHUSDT","ZECUSDT","XTZUSDT","BNBUSDT","ATOMUSDT","ONTUSDT","IOTAUSDT","BATUSDT","VETUSDT","NEOUSDT","QTUMUSDT","IOSTUSDT","THETAUSDT","ALGOUSDT","ZILUSDT","KNCUSDT","ZRXUSDT","COMPUSDT","DOGEUSDT","SXPUSDT","KAVAUSDT","BANDUSDT","RLCUSDT","SNXUSDT","DOTUSDT","YFIUSDT","CRVUSDT","TRBUSDT","RUNEUSDT","SUSHIUSDT","EGLDUSDT","SOLUSDT","ICXUSDT","STORJUSDT","UNIUSDT","AVAXUSDT","ENJUSDT","FLMUSDT","KSMUSDT","NEARUSDT","AAVEUSDT","FILUSDT","RSRUSDT","LRCUSDT","BELUSDT","AXSUSDT","ZENUSDT","SKLUSDT","GRTUSDT","1INCHUSDT","SANDUSDT","CHZUSDT","ANKRUSDT","RVNUSDT","SFPUSDT","COTIUSDT","CHRUSDT","MANAUSDT","ALICEUSDT","GTCUSDT","HBARUSDT","ONEUSDT","DENTUSDT","CELRUSDT","HOTUSDT","MTLUSDT","OGNUSDT","NKNUSDT","1000SHIBUSDT","BAKEUSDT","BTCDOMUSDT","MASKUSDT","ICPUSDT","IOTXUSDT","C98USDT","ATAUSDT","DYDXUSDT","1000XECUSDT","GALAUSDT","CELOUSDT","ARUSDT","ARPAUSDT","CTSIUSDT","LPTUSDT","ENSUSDT","PEOPLEUSDT","ROSEUSDT","DUSKUSDT","FLOWUSDT","IMXUSDT"]
    crypto_metric_list = []
    file_dir = '/Users/aming/project/python/binance-public-data-master/python/data/futures/um/monthly/klines/ETHUSDT/15m/2025-06-01_2025-12-01/ETHUSDT-15m-2025-06.zip'
    for crypto in crypto_list:
        # crypto = "BNBUSDT"
        # crypto_metric[crypto] = {}
        df_price = load_daily_data(file_dir=file_dir, start_date=start_date, end_date=end_date, interval="15m", crypto=crypto)
        strategy = MAStrategyAnalyzer(
            df_price, 
            short_window=5, 
            long_window=20,
            crypto=crypto,
            commission_rate=0.0001  # 手续费率：0.01%
        )
        # 如需保存图表，取消下面一行的注释并指定路径
        other_metrics = strategy.run_strategy(save_plot_path=f"{save_path}/strategy_results_with_commission_{crypto}.png")
        crypto_metric_list.append(other_metrics)
        # crypto_metric[crypto] = other_metrics
    print(crypto_metric_list)