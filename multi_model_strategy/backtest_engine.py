"""
回测引擎模块
负责模拟真实交易、计算绩效指标
"""
import numpy as np
import pandas as pd


class BacktestEngine:
    """
    回测引擎：真实交易模拟器
    """
    
    def __init__(self, open_train, close_train, open_test, close_test, fees_rate=0.0005, annual_bars=35040):
        """
        Args:
            open_train (np.ndarray): 训练集开盘价
            close_train (np.ndarray): 训练集收盘价
            open_test (np.ndarray): 测试集开盘价
            close_test (np.ndarray): 测试集收盘价
            fees_rate (float): 手续费率
            annual_bars (int): 年化 bar 数
        """
        self.open_train = open_train
        self.close_train = close_train
        self.open_test = open_test
        self.close_test = close_test
        self.fees_rate = fees_rate
        self.annual_bars = annual_bars
        
        self.backtest_results = {}
    
    def run_backtest(self, pos, data_range='test'):
        """
        模拟真实交易场景
        
        Args:
            pos (np.ndarray): 仓位序列
            data_range (str): 'train' 或 'test'
        
        Returns:
            tuple: (pnl, metrics)
        """
        # 获取对应的价格数据
        if data_range == 'train':
            open_data = self.open_train
            close_data = self.close_train
        elif data_range == 'test':
            open_data = self.open_test
            close_data = self.close_test
        else:
            raise ValueError(f"不支持的data_range: {data_range}")
        
        # 转换为 numpy 数组
        if isinstance(open_data, pd.Series):
            open_data = open_data.values
        if isinstance(close_data, pd.Series):
            close_data = close_data.values
        
        pos = np.asarray(pos).flatten()
        
        # 确保长度匹配
        min_len = min(len(pos), len(open_data), len(close_data))
        pos = pos[:min_len]
        open_data = open_data[:min_len]
        close_data = close_data[:min_len]
        
        next_open = np.concatenate((open_data[1:], np.array([close_data[-1]])))
        close = close_data
        
        real_pos = pos
        pos_change = np.concatenate((np.array([0]), np.diff(real_pos)))
        
        # 决定交易价格（模拟滑点）
        which_price_to_trade = np.where(
            pos_change > 0,
            np.maximum(close, next_open),  # 买入用更高价格
            np.where(
                pos_change < 0,
                np.minimum(close, next_open),  # 卖出用更低价格
                close
            )
        )
        
        next_trade_close = np.concatenate((which_price_to_trade[1:], np.array([which_price_to_trade[-1]])))
        rets = np.log(next_trade_close) - np.log(which_price_to_trade)
        
        # 计算收益（扣除手续费）
        gain_loss = real_pos * rets - abs(pos_change) * self.fees_rate
        pnl = gain_loss.cumsum()
        
        # 计算性能指标
        metrics = self._calculate_metrics(gain_loss, pnl)
        
        return pnl, metrics
    
    def _calculate_metrics(self, gain_loss, pnl):
        """计算绩效指标"""
        win_rate_bar = np.sum(gain_loss > 0) / len(gain_loss) if len(gain_loss) > 0 else 0
        avg_gain_bar = np.mean(gain_loss[gain_loss > 0]) if np.any(gain_loss > 0) else 0
        avg_loss_bar = np.abs(np.mean(gain_loss[gain_loss < 0])) if np.any(gain_loss < 0) else 0
        profit_loss_ratio_bar = avg_gain_bar / avg_loss_bar if avg_loss_bar != 0 else np.inf
        
        annual_return = np.mean(gain_loss) * self.annual_bars
        sharpe_ratio = annual_return / (np.std(gain_loss) * np.sqrt(self.annual_bars)) if np.std(gain_loss) > 0 else 0
        
        # 最大回撤
        peak_values = np.maximum.accumulate(pnl)
        peak_values = np.where(peak_values == 0, 1.0, peak_values)
        drawdowns = (pnl - peak_values) / np.abs(peak_values)
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # 卡尔玛比率
        if max_drawdown < -0.0001:
            Calmar_Ratio = annual_return / abs(max_drawdown)
        else:
            Calmar_Ratio = np.inf if annual_return > 0 else 0
        
        metrics = {
            "Win Rate": win_rate_bar,
            "Profit/Loss Ratio": profit_loss_ratio_bar,
            "Annual Return": annual_return,
            "MAX_Drawdown": max_drawdown,
            "Sharpe Ratio": sharpe_ratio,
            "Calmar Ratio": Calmar_Ratio
        }
        
        return metrics
    
    def backtest_all_models(self, predictions):
        """
        回测所有模型
        
        Args:
            predictions (dict): 所有模型的预测结果
        
        Returns:
            dict: 回测结果
        """
        print("正在回测所有模型...")
        
        for model_name, pred in predictions.items():
            print(f"回测 {model_name} 模型...")
            
            train_pos = pred['train']
            test_pos = pred['test']
            
            train_pnl, train_metrics = self.run_backtest(train_pos, 'train')
            test_pnl, test_metrics = self.run_backtest(test_pos, 'test')
            
            self.backtest_results[model_name] = {
                'train_pnl': train_pnl,
                'test_pnl': test_pnl,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics
            }
            
            print(f"  样本内夏普: {train_metrics['Sharpe Ratio']:.4f}")
            print(f"  样本外夏普: {test_metrics['Sharpe Ratio']:.4f}")
        
        return self.backtest_results
    
    def get_performance_summary(self):
        """获取所有模型的绩效汇总"""
        if not self.backtest_results:
            print("请先运行回测")
            return None
        
        summary_data = []
        
        for model_name, results in self.backtest_results.items():
            train_metrics = results['train_metrics']
            test_metrics = results['test_metrics']
            
            summary_data.append({
                '模型': model_name,
                '样本内年化收益': train_metrics['Annual Return'],
                '样本内夏普比率': train_metrics['Sharpe Ratio'],
                '样本内最大回撤': train_metrics['MAX_Drawdown'],
                '样本外年化收益': test_metrics['Annual Return'],
                '样本外夏普比率': test_metrics['Sharpe Ratio'],
                '样本外最大回撤': test_metrics['MAX_Drawdown']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.round(4)
        
        print("\n模型绩效汇总:")
        print(summary_df.to_string(index=False))
        
        return summary_df

