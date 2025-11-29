"""
可视化模块
负责绘制回测结果、Regime/Risk 诊断图
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Visualizer:
    """
    可视化工具：回测结果、Regime/Risk 诊断
    """
    
    def __init__(self, z_index, close_train, close_test):
        """
        Args:
            z_index (np.ndarray): 时间索引
            close_train (np.ndarray): 训练集收盘价
            close_test (np.ndarray): 测试集收盘价
        """
        self.z_index = z_index
        self.close_train = close_train
        self.close_test = close_test
    
    def plot_backtest_results(self, backtest_results, model_name='Ensemble', ensemble_weights=None):
        """
        绘制回测结果
        
        Args:
            backtest_results (dict): 回测结果
            model_name (str): 要绘制的模型名称
            ensemble_weights (dict, optional): 集成模型权重
        """
        if model_name not in backtest_results:
            print(f"模型 {model_name} 的回测结果不存在")
            return
        
        result = backtest_results[model_name]
        train_pnl = result['train_pnl']
        test_pnl = result['test_pnl']
        train_metrics = result['train_metrics']
        test_metrics = result['test_metrics']
        
        # 获取时间索引
        train_index = self.z_index[:len(train_pnl)]
        test_index = self.z_index[len(train_pnl):len(train_pnl) + len(test_pnl)]
        
        # 创建图表
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        
        # 根据模型类型设置标题
        if model_name == 'Ensemble':
            title = f'多模型组合 (Ensemble) 回测结果'
            color = 'red'
        else:
            title = f'{model_name} 回测结果'
            color = 'green'
        
        fig.suptitle(title, fontsize=18, fontweight='bold')
        
        # 训练集价格
        axs[0, 0].plot(train_index, self.close_train, 'b-', linewidth=1.5)
        axs[0, 0].set_title('训练集价格', fontsize=12, fontweight='bold')
        axs[0, 0].set_ylabel('价格', fontsize=10)
        axs[0, 0].grid(True, alpha=0.3, linestyle='--')
        
        # 训练集PnL
        axs[1, 0].plot(train_index, train_pnl, 'g-', linewidth=2)
        axs[1, 0].set_title('训练集累计PnL', fontsize=12, fontweight='bold')
        axs[1, 0].set_ylabel('累计收益', fontsize=10)
        axs[1, 0].grid(True, alpha=0.3, linestyle='--')
        axs[1, 0].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # 添加训练集指标
        train_text = (f"年化收益: {train_metrics['Annual Return']:.2%}\n"
                     f"夏普比率: {train_metrics['Sharpe Ratio']:.3f}\n"
                     f"最大回撤: {train_metrics['MAX_Drawdown']:.2%}")
        axs[1, 0].text(0.02, 0.98, train_text,
                      transform=axs[1, 0].transAxes,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6),
                      fontsize=9)
        
        # 测试集价格
        axs[0, 1].plot(test_index, self.close_test, 'b-', linewidth=1.5)
        axs[0, 1].set_title('测试集价格', fontsize=12, fontweight='bold')
        axs[0, 1].set_ylabel('价格', fontsize=10)
        axs[0, 1].grid(True, alpha=0.3, linestyle='--')
        
        # 测试集PnL
        axs[1, 1].plot(test_index, test_pnl, color=color, linewidth=2.5)
        axs[1, 1].set_title('测试集累计PnL', fontsize=12, fontweight='bold')
        axs[1, 1].set_ylabel('累计收益', fontsize=10)
        axs[1, 1].grid(True, alpha=0.3, linestyle='--')
        axs[1, 1].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # 添加测试集指标
        if model_name == 'Ensemble' and ensemble_weights:
            weight_info = "\n".join([f"{name}: {w:.1%}" for name, w in ensemble_weights.items()])
            metrics_text = (f"年化收益: {test_metrics['Annual Return']:.2%}\n"
                          f"夏普比率: {test_metrics['Sharpe Ratio']:.3f}\n"
                          f"最大回撤: {test_metrics['MAX_Drawdown']:.2%}\n"
                          f"卡尔玛比率: {test_metrics['Calmar Ratio']:.3f}\n"
                          f"胜率: {test_metrics['Win Rate']:.2%}\n"
                          f"\n模型权重:\n{weight_info}")
        else:
            metrics_text = (f"年化收益: {test_metrics['Annual Return']:.2%}\n"
                          f"夏普比率: {test_metrics['Sharpe Ratio']:.3f}\n"
                          f"最大回撤: {test_metrics['MAX_Drawdown']:.2%}\n"
                          f"卡尔玛比率: {test_metrics['Calmar Ratio']:.3f}\n"
                          f"胜率: {test_metrics['Win Rate']:.2%}\n"
                          f"盈亏比: {test_metrics['Profit/Loss Ratio']:.3f}")
        
        axs[1, 1].text(0.02, 0.98, metrics_text,
                      transform=axs[1, 1].transAxes,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
                      fontsize=9)
        
        plt.tight_layout()
        plt.show(block=False)
    
    def plot_regime_and_risk_scalers(self, predictions, regime_scaler_train, regime_scaler_test,
                                     risk_scaler_train, risk_scaler_test, model_name='Ensemble'):
        """
        绘制 Regime & Risk 缩放因子诊断图
        
        Args:
            predictions (dict): 模型预测结果
            regime_scaler_train (np.ndarray): 训练集 Regime 缩放因子
            regime_scaler_test (np.ndarray): 测试集 Regime 缩放因子
            risk_scaler_train (np.ndarray): 训练集 Risk 缩放因子
            risk_scaler_test (np.ndarray): 测试集 Risk 缩放因子
            model_name (str): 模型名称
        """
        if model_name not in predictions:
            print(f"模型 {model_name} 的预测结果不存在")
            return
        
        # 拼接训练/测试段
        train_pos = np.asarray(predictions[model_name]['train']).flatten()
        test_pos = np.asarray(predictions[model_name]['test']).flatten()
        pos_all = np.concatenate([train_pos, test_pos])
        
        regime_all = np.concatenate([regime_scaler_train, regime_scaler_test])
        risk_all = np.concatenate([risk_scaler_train, risk_scaler_test])
        
        # 时间索引与价格
        idx = pd.to_datetime(self.z_index[:len(pos_all)])
        close_all = np.concatenate([self.close_train, self.close_test])[:len(pos_all)]
        
        fig, axs = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        fig.suptitle(f'Regime & Risk 缩放诊断 - {model_name}', fontsize=16, fontweight='bold')
        
        # 1) 价格 + 仓位
        ax1 = axs[0]
        ax1.plot(idx, close_all, 'b-', linewidth=1.5, label='Price')
        ax1.set_ylabel('Price', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(idx, pos_all, 'r-', linewidth=1.0, alpha=0.8, label='Position')
        ax1_twin.set_ylabel('Position', fontsize=10)
        ax1.set_title('价格与仓位（已包含 Regime & Risk 缩放后）', fontsize=12, fontweight='bold')
        
        # 2) Regime 缩放
        ax2 = axs[1]
        ax2.plot(idx, regime_all, 'g-', linewidth=1.2)
        ax2.set_ylabel('Regime scaler', fontsize=10)
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_title('Regime 缩放因子', fontsize=12, fontweight='bold')
        
        # 3) Risk 缩放
        ax3 = axs[2]
        ax3.plot(idx, risk_all, 'm-', linewidth=1.2)
        ax3.set_ylabel('Risk scaler', fontsize=10)
        ax3.set_ylim(-0.05, 1.05)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_title('风控 & 拥挤度 缩放因子', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show(block=False)

