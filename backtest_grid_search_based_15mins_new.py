from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math
import matplotlib.pyplot as plt

def generate_etime_close_data_divd_time(bgn_date, end_date, index_code, frequency):
    """生成行情数据"""
    read_file_path = '/Users/aming/project/python/课程代码合集/单因子评价方式_2/' + index_code + '_' + frequency + '.xlsx'
    kbars = pd.read_excel(read_file_path)
    
    kbars['tdate'] = pd.to_datetime(kbars['etime']).dt.date
    dt = pd.to_datetime(kbars['etime'], format='%Y-%m-%d %H:%M:%S.%f')
    kbars['etime'] = pd.Series([pd.Timestamp(x).round('s').to_pydatetime() for x in dt])
    
    # 根据日期范围筛选数据
    mask = (pd.to_datetime(kbars['etime']) >= pd.to_datetime(bgn_date)) & \
           (pd.to_datetime(kbars['etime']) <= pd.to_datetime(end_date))
    kbars = kbars[mask].reset_index(drop=True)
    
    return kbars[['etime', 'tdate', 'close']].reset_index(drop=True)

def calculate_positions(predictions, scale=0.0005, position_size=1):
    """计算持仓信号 sigmoid  tanh, relu, zscore, """
    positions = (predictions / scale) * position_size
    return np.clip(positions, -1, 1)

# 时间轴:     t0 ————→ t1 ————→ t2 ————→ t3 ————→ t4
# 持仓:       0.5      0.8     -0.2      0.3      0.1
# 价格变化:      ↑2%      ↓1%      ↑2%      ↑2%
# 策略收益:            1%      -0.8%    -0.4%     0.6%

# 对应关系:
# t0持仓(0.5) × t0→t1收益(2%) = t1策略收益(1%)
# t1持仓(0.8) × t1→t2收益(-1%) = t2策略收益(-0.8%)
# t2持仓(-0.2) × t2→t3收益(2%) = t3策略收益(-0.4%)
# t3持仓(0.3) × t3→t4收益(2%) = t4策略收益(0.6%)


# t0时刻：
# - 收到信号：下一期应该持仓0.5
# - 执行交易：买入0.5的仓位
# - 持仓状态：0.5

# t0→t1期间：
# - 持仓0.5不变
# - 价格变化2%
# - 策略收益：0.5 × 2% = 1%

# t1时刻：
# - 收到新信号：下一期应该持仓0.8  
# - 执行交易：从0.5调整到0.8（增加0.3）
# - 持仓状态：0.8

# t1→t2期间：
# - 持仓0.8不变
# - 价格变化-1%
# - 策略收益：0.8 × (-1%) = -0.8%

def calculate_returns_and_nav(data, positions):
    """计算收益率和净值"""
    price_changes = data['close'].pct_change().fillna(0)
    strategy_returns = positions[:-1] * price_changes[1:]
    # position_diff = np.abs(positions[1:] - positions[:-1]) 80% -10%  90%
    # pnl = strategy_returns
    # net_pnl = pnl - fees*position_diff
    # cumu_cost = np.cumsum(fees*position_diff)
    cumulative_returns = (1 + strategy_returns).cumprod() # 注意这里，可以修正为单利模式，和加入手续费
    # cumulative_returns = 1 + strategy_returns.cumsum() # 注意这里，可以修正为单利模式，和加入手续费
    return strategy_returns, cumulative_returns

def calculate_metrics_by_period(data, positions, returns, nav):
   """计算给定时期的评价指标"""
   metrics = {}
   rf = 0.03
    # param_metric = weights2calculate([0.5,0.2, 0.3])
   
   # 基础指标计算
   annual_return = (nav.iloc[-1] ** (252/len(nav))) - 1
   annual_vol = returns.std() * np.sqrt(252)
   sharpe = (annual_return - rf) / annual_vol if annual_vol != 0 else 0
   

   # 计算最大回撤及其起始日期
#    回撤 = 历史最高净值 - 当前净值
   drawdown = nav.cummax() - nav
#    最大回撤率 = 最大回撤金额 / 历史最高净值
   max_drawdown = drawdown.max() / nav.cummax().max()
   max_drawdown_end_idx = drawdown.argmax()
   max_drawdown_start_idx = nav[:max_drawdown_end_idx+1].argmax()
   
   max_drawdown_start_date = data['tdate'].iloc[max_drawdown_start_idx]
   max_drawdown_end_date = data['tdate'].iloc[max_drawdown_end_idx]
   
   # 计算胜率和盈亏比
   daily_pnl = returns * positions[:-1]
   win_rate = (daily_pnl > 0).mean()

   # 盈亏比 = 盈利的平均值 / 亏损的平均值
   gain_loss_ratio = abs(daily_pnl[daily_pnl > 0].mean() / daily_pnl[daily_pnl < 0].mean()) \
                    if len(daily_pnl[daily_pnl < 0]) > 0 else np.inf
#    防止除零错误：如果没有亏损交易日，返回无穷大
   # 衡量因子的择时有效性，或者因子的稳定性，当前bar对应的，过去100个bar，他的gain_loss_ratio，

   metrics.update({
       '总收益': nav.iloc[-1] - 1,
       '年化收益': annual_return,
       '年化波动率': annual_vol,
       '夏普比率': sharpe,
       '最大回撤率': max_drawdown,
       '最大回撤起始日': max_drawdown_start_date,
       '最大回撤结束日': max_drawdown_end_date,
       '总交易次数': len(returns),
       '胜率': win_rate,
       '盈亏比': gain_loss_ratio
   })
   
   return metrics



def backtest(original_data, index_code, frequency, n_days):
    """回测主函数"""
    # 数据预处理
    final_frame = original_data[['tdate', 'etime', 'close', 'fct']].dropna(axis=0).reset_index(drop=True)
    # 如果应对多个品种的计算，可以用下面的方式。
    # raw_data, fct_data = {},{}
    # raw_data['btc_30m'] = {} 
    # fct_data['btc_30m'] = {} 
    
    # 计算时间步长和收益率
    t_delta = int(1 * n_days) if frequency == '15' else int(int(240 / int(frequency)) * n_days)
    final_frame['ret'] = final_frame['close'].shift(-t_delta) / final_frame['close'] - 1 # 这里的return，是我们的label，是未来t_delta天的收益率，他和我们计算绩效的时候，不一样。
    final_frame = final_frame.dropna(axis=0).reset_index(drop=True)

    # 训练测试集划分
    train_set_end_index = final_frame[(final_frame['etime'].dt.year == 2019) & 
                                    (final_frame['etime'].dt.month == 12) & 
                                    (final_frame['etime'].dt.day == 31)].index[0]
    
    # 准备训练数据
    X_train = final_frame.loc[:train_set_end_index, 'fct'].values.reshape(-1, 1)
    y_train = final_frame.loc[:train_set_end_index, 'ret'].values.reshape(-1, 1)
    X_test = final_frame.loc[train_set_end_index+1:, 'fct'].values.reshape(-1, 1)

    # 模型训练和预测 Y = AX + B
    model = LinearRegression(fit_intercept=True) # t-statistic，weight，bias
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train).flatten()
    y_test_pred = model.predict(X_test).flatten()

    # 初始化结果DataFrame
    indicators_frame = pd.DataFrame()
    
    # 计算每年的指标
    for year in final_frame['etime'].dt.year.unique():
        year_mask = final_frame['etime'].dt.year == year
        year_data = final_frame[year_mask]
        
        if len(year_data) > 0:
            year_positions = calculate_positions(
                model.predict(year_data['fct'].values.reshape(-1, 1)).flatten()
            )
            year_returns, year_nav = calculate_returns_and_nav(year_data, year_positions)
            year_metrics = calculate_metrics_by_period(
                year_data, year_positions, year_returns, year_nav
            )
            indicators_frame = pd.concat([
                indicators_frame, 
                pd.DataFrame(year_metrics, index=[year])
            ])
    
    # 计算样本内外指标
    train_data = final_frame[:train_set_end_index+1]
    test_data = final_frame[train_set_end_index+1:]
    
    # 样本内指标
    train_positions = calculate_positions(y_train_pred)
    train_returns, train_nav = calculate_returns_and_nav(train_data, train_positions)
    indicators_frame.loc['样本内'] = calculate_metrics_by_period(
        train_data, train_positions, train_returns, train_nav
    )
    
    # 样本外指标
    test_positions = calculate_positions(y_test_pred)
    test_returns, test_nav = calculate_returns_and_nav(test_data, test_positions)
    indicators_frame.loc['样本外'] = calculate_metrics_by_period(
        test_data, test_positions, test_returns, test_nav
    )
    
    # 总体指标
    total_positions = np.concatenate([train_positions, test_positions])
    total_returns, total_nav = calculate_returns_and_nav(final_frame, total_positions)
    indicators_frame.loc['总体'] = calculate_metrics_by_period(
        final_frame, total_positions, total_returns, total_nav
    )
    
#     年度计算：时间维度的策略分析
# 样本内外计算：模型质量的技术分析

    return indicators_frame