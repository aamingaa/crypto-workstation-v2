"""
Multi-Model Quantitative Trading Strategy Package

多模型量化交易策略包（整合GP因子）

主要功能：
- GP 因子评估与筛选
- 多模型训练（OLS, Ridge, Lasso, XGBoost, LightGBM）
- Regime + Risk 层仓位缩放
- Kelly bet sizing（可选）
- Triple Barrier 标注（可选）
- 回测与诊断工具
"""

from .strategy import QuantTradingStrategy
from .config import StrategyConfig, DataConfig
from .data_module import DataModule
from .factor_engine import FactorEngine
from .alpha_models import AlphaModelTrainer
from .position_scaling import (
    RegimeScaler,
    RiskScaler,
    KellyBetSizer,
    PositionScalingManager
)
from .backtest_engine import BacktestEngine
from .visualization import Visualizer
from .diagnostics import DiagnosticTools

__version__ = '1.0.0'

__all__ = [
    # 主策略类
    'QuantTradingStrategy',
    
    # 配置
    'StrategyConfig',
    'DataConfig',
    
    # 核心模块
    'DataModule',
    'FactorEngine',
    'AlphaModelTrainer',
    'BacktestEngine',
    
    # 仓位管理
    'RegimeScaler',
    'RiskScaler',
    'KellyBetSizer',
    'PositionScalingManager',
    
    # 工具
    'Visualizer',
    'DiagnosticTools',
]


# 便捷创建函数（快速上手）
def create_strategy_from_expressions(factor_expressions, sym='ETHUSDT',
                                     train_dates=('2025-01-01', '2025-03-01'),
                                     test_dates=('2025-03-01', '2025-04-01'),
                                     **kwargs):
    """
    快速创建策略（无需YAML配置）
    
    Args:
        factor_expressions (list): 因子表达式列表
        sym (str): 交易对
        train_dates (tuple): 训练日期范围
        test_dates (tuple): 测试日期范围
        **kwargs: 其他配置参数
    
    Returns:
        QuantTradingStrategy: 策略实例
    
    Example:
        >>> from multi_model_strategy import create_strategy_from_expressions
        >>> 
        >>> factors = ['ta_rsi_14(close)', 'ta_ema_20(close)']
        >>> strategy = create_strategy_from_expressions(
        ...     factors,
        ...     sym='ETHUSDT',
        ...     train_dates=('2025-01-01', '2025-02-01'),
        ...     test_dates=('2025-02-01', '2025-03-01')
        ... )
        >>> strategy.run_full_pipeline()
        >>> strategy.plot_results()
    """
    return QuantTradingStrategy.from_expressions_simple(
        factor_expressions, sym, train_dates, test_dates, **kwargs
    )


def create_strategy_from_yaml(yaml_path, factor_expressions, **kwargs):
    """
    从YAML配置 + 因子表达式创建策略
    
    Args:
        yaml_path (str): YAML配置文件路径
        factor_expressions (list): 因子表达式列表
        **kwargs: 额外策略配置
    
    Returns:
        QuantTradingStrategy: 策略实例
    
    Example:
        >>> from multi_model_strategy import create_strategy_from_yaml
        >>> 
        >>> factors = ['ta_rsi_14(close)']
        >>> strategy = create_strategy_from_yaml(
        ...     'config.yaml',
        ...     factors,
        ...     max_factors=5
        ... )
        >>> strategy.run_full_pipeline()
    """
    return QuantTradingStrategy.from_yaml_with_expressions(
        yaml_path, factor_expressions, strategy_config=kwargs
    )

