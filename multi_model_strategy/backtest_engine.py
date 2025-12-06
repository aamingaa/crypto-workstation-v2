"""
回测引擎模块
负责模拟真实交易、计算绩效指标
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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

    def run_backtest_subperiod(self, pos, data_range='test', start_bar=None, end_bar=None):
        """
        小区间回测：在指定 bar 区间内重新计算 PnL 与绩效
        
        Args:
            pos (np.ndarray): 全段仓位序列（train 或 test）
            data_range (str): 'train' 或 'test'
            start_bar (int, optional): 区间起始 bar（相对于当前 data_range 的 0-based 索引，含）
            end_bar (int, optional): 区间结束 bar（相对于当前 data_range 的 0-based 索引，不含）
        
        Returns:
            tuple: (pnl_sub, metrics_sub)
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
        
        # 统一长度
        min_len = min(len(pos), len(open_data), len(close_data))
        pos = pos[:min_len]
        open_data = open_data[:min_len]
        close_data = close_data[:min_len]
        
        # 处理子区间索引
        if start_bar is None:
            start_bar = 0
        if end_bar is None or end_bar > min_len:
            end_bar = min_len
        if start_bar < 0 or start_bar >= end_bar:
            raise ValueError(f"非法的小区间索引: start_bar={start_bar}, end_bar={end_bar}, 长度={min_len}")
        
        pos_sub = pos[start_bar:end_bar]
        open_sub = open_data[start_bar:end_bar]
        close_sub = close_data[start_bar:end_bar]
        
        # 与 run_backtest 中相同的交易逻辑
        next_open = np.concatenate((open_sub[1:], np.array([close_sub[-1]])))
        close = close_sub
        
        real_pos = pos_sub
        pos_change = np.concatenate((np.array([0]), np.diff(real_pos)))
        
        which_price_to_trade = np.where(
            pos_change > 0,
            np.maximum(close, next_open),
            np.where(
                pos_change < 0,
                np.minimum(close, next_open),
                close
            )
        )
        
        next_trade_close = np.concatenate((which_price_to_trade[1:], np.array([which_price_to_trade[-1]])))
        rets = np.log(next_trade_close) - np.log(which_price_to_trade)
        
        gain_loss = real_pos * rets - abs(pos_change) * self.fees_rate
        pnl = gain_loss.cumsum()
        
        metrics = self._calculate_metrics(gain_loss, pnl)
        return pnl, metrics
    
    def _calculate_metrics(self, gain_loss, pnl):
        """计算绩效指标（基于 log return）"""
        gain_loss = np.asarray(gain_loss)
        pnl = np.asarray(pnl)

        # 单bar胜率与盈亏比
        win_rate_bar = np.sum(gain_loss > 0) / len(gain_loss) if len(gain_loss) > 0 else 0
        avg_gain_bar = np.mean(gain_loss[gain_loss > 0]) if np.any(gain_loss > 0) else 0
        avg_loss_bar = np.abs(np.mean(gain_loss[gain_loss < 0])) if np.any(gain_loss < 0) else 0
        profit_loss_ratio_bar = avg_gain_bar / avg_loss_bar if avg_loss_bar != 0 else np.inf
        
        # 年化收益与夏普（gain_loss 视为每bar的 log return）
        mean_ret = np.mean(gain_loss) if len(gain_loss) > 0 else 0.0
        std_ret = np.std(gain_loss) if len(gain_loss) > 1 else 0.0
        annual_return = mean_ret * self.annual_bars
        sharpe_ratio = annual_return / (std_ret * np.sqrt(self.annual_bars)) if std_ret > 0 else 0.0
        
        # 将累积 log 收益还原为净值曲线（初始净值为 1）
        equity = np.exp(pnl)
        peak_equity = np.maximum.accumulate(equity)
        # 避免除零
        peak_equity = np.where(peak_equity == 0, 1.0, peak_equity)
        drawdowns = equity / peak_equity - 1.0
        max_drawdown = drawdowns.min() if len(drawdowns) > 0 else 0.0
        
        # Calmar Ratio：用年化 log return / |maxDD|
        if max_drawdown < -0.0001:
            calmar_ratio = annual_return / abs(max_drawdown)
        else:
            calmar_ratio = np.inf if annual_return > 0 else 0.0
        
        metrics = {
            "Win Rate": win_rate_bar,
            "Profit/Loss Ratio": profit_loss_ratio_bar,
            "Annual Return": annual_return,
            "MAX_Drawdown": max_drawdown,
            "Sharpe Ratio": sharpe_ratio,
            "Calmar Ratio": calmar_ratio
        }
        
        return metrics
    
    def backtest_with_backtest_engine(self, pos_test, pos_train=None, fees_rate=None):
        """
        使用 multi_model_strategy.BacktestEngine 对 GP 产生的仓位进行统一口径回测
        
        Args:
            pos_test (np.ndarray): 测试集仓位序列
            pos_train (np.ndarray or None): 训练集仓位序列；若为 None，则不回测训练段
            fees_rate (float or None): 手续费率；None 时优先使用 self.fees_rate，否则回退到 0.0005
        
        Returns:
            dict: {
                "train": {"pnl": np.ndarray, "metrics": dict} or None,
                "test":  {"pnl": np.ndarray, "metrics": dict}
            }
        """
        # 手续费率与年化 bar 数与 multi_model_strategy 保持一致口径
        # if fees_rate is None:
        #     fees_rate = getattr(self, "fees_rate", 0.0005)
        # annual_bars = getattr(self, "annual_bars", calculate_annual_bars(self.freq))

        # engine = BacktestEngine(
        #     self.open_train,
        #     self.close_train,
        #     self.open_test,
        #     self.close_test,
        #     fees_rate=fees_rate,
        #     annual_bars=annual_bars,
        # )

        results = {"train": None, "test": None}

        # 测试集回测（必跑）
        pnl_test, metrics_test = self.run_backtest(pos_test, data_range="test")
        results["test"] = {"pnl": pnl_test, "metrics": metrics_test}

        # 训练集回测（可选）
        if pos_train is not None:
            pnl_train, metrics_train = self.run_backtest(pos_train, data_range="train")
            results["train"] = {"pnl": pnl_train, "metrics": metrics_train}

        return results

    def real_trading_simulation_plot(self, pos_test, pos_train=None, save_dir=None, title=None):
        """
        参考 GPAnalyzer.real_trading_simulation_plot 的接口，
        使用 BacktestEngine 的交易逻辑绘制 train/test 价格 + 净值曲线，并在图中标注测试集绩效指标。
        
        Args:
            pos_test (np.ndarray): 测试集仓位
            pos_train (np.ndarray, optional): 训练集仓位；若为 None，则不绘制训练段 PnL
            save_dir (str or Path, optional): 图片保存目录；None 则不保存，仅展示
            title (str, optional): 总标题
        Returns:
            dict: {"train": {...} or None, "test": {...}}，包含 pnl 与指标
        """
        results = {"train": None, "test": None}

        # 回测测试集
        pnl_test, metrics_test = self.run_backtest(pos_test, data_range="test")
        results["test"] = {"pnl": pnl_test, "metrics": metrics_test}

        # 回测训练集（可选）
        if pos_train is not None:
            pnl_train, metrics_train = self.run_backtest(pos_train, data_range="train")
            results["train"] = {"pnl": pnl_train, "metrics": metrics_train}
        else:
            pnl_train, metrics_train = None, None

        # 价格与索引（尽量复用 Series 的时间索引）
        close_train = self.close_train
        close_test = self.close_test

        if isinstance(close_train, pd.Series):
            idx_train = close_train.index
            price_train = close_train.values
        else:
            price_train = np.asarray(close_train)
            idx_train = np.arange(len(price_train))

        if isinstance(close_test, pd.Series):
            idx_test = close_test.index
            price_test = close_test.values
        else:
            price_test = np.asarray(close_test)
            idx_test = np.arange(len(price_test))

        # 将 pnl 视为累计 log 收益，直接画形态（与 GPAnalyzer 保持一致）
        fig, axs = plt.subplots(2, 2, figsize=(10, 12))
        if title is None:
            title = "Real Trading Simulation (BacktestEngine)"
        fig.suptitle(title, fontsize=16)

        # Train price
        axs[0, 0].plot(idx_train, price_train, 'b-')
        axs[0, 0].set_title('price_train')
        if isinstance(idx_train, pd.DatetimeIndex):
            axs[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            axs[0, 0].xaxis.set_major_locator(mdates.MonthLocator())

        # Train pnl
        if pnl_train is not None:
            axs[1, 0].plot(idx_train[:len(pnl_train)], pnl_train, 'r-')
        axs[1, 0].set_title('pnl_train')
        if isinstance(idx_train, pd.DatetimeIndex):
            axs[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            axs[1, 0].xaxis.set_major_locator(mdates.MonthLocator())

        # Test price
        axs[0, 1].plot(idx_test, price_test, 'b-')
        axs[0, 1].set_title('price_test')
        if isinstance(idx_test, pd.DatetimeIndex):
            axs[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            axs[0, 1].xaxis.set_major_locator(mdates.MonthLocator())

        # Test pnl
        axs[1, 1].plot(idx_test[:len(pnl_test)], pnl_test, 'r-')
        axs[1, 1].set_title('pnl_test')
        if isinstance(idx_test, pd.DatetimeIndex):
            axs[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            axs[1, 1].xaxis.set_major_locator(mdates.MonthLocator())

        # 在测试集 pnl 子图上标注指标
        annotation_text = "\n".join([f"{key}: {value:.4f}" for key, value in metrics_test.items()])
        axs[1, 1].annotate(
            annotation_text,
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )

        for ax in axs.flat:
            ax.grid(True)

        plt.tight_layout()

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / "net_value_performance.png"
            plt.savefig(out_path)
        else:
            plt.show()

        plt.close(fig)
        return results
    
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

