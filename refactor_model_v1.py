"""
使用示例：展示如何使用新的模块化策略框架

三种使用方式：
1. 快速创建（无需YAML）
2. 从YAML + 因子表达式创建
3. 从YAML + CSV文件创建
"""
import sys
from pathlib import Path

# 确保项目路径和 gp_crypto_next 都在 sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

gp_crypto_dir = project_root / "gp_crypto_next"
if str(gp_crypto_dir) not in sys.path:
    sys.path.insert(0, str(gp_crypto_dir))

import matplotlib.pyplot as plt

# ========== 方式1: 快速创建（推荐用于快速测试） ==========
from multi_model_strategy import create_strategy_from_expressions
from multi_model_strategy import create_strategy_from_yaml


# 定义因子表达式
factor_expressions = [
    'ta_dema_55(ta_dx_25(neg(h), ta_trix_8(h), ta_trix_21(ori_trix_21)))',
    'ta_lr_slope_20(ts_delta_17(ori_ta_macd))',
]

# 创建策略
# strategy = create_strategy_from_expressions(
#     factor_expressions=factor_expressions,
#     sym='ETHUSDT',
#     train_dates=('2025-01-01', '2025-01-20'),
#     test_dates=('2025-01-20', '2025-01-31'),
#     max_factors=5,
#     fees_rate=0.0005,
#     # 数据源配置
#     data_source='coarse_grain',  # 或 'kline'
#     coarse_grain_period='2h',
#     feature_lookback_bars=8,
# )


yaml_path = 'gp_crypto_next/coarse_grain_parameters.yaml'

strategy = create_strategy_from_yaml(
    yaml_path=yaml_path,
    factor_expressions=factor_expressions,
    kwargs=None
)



# from multi_model_strategy import (
#     DataModule,
#     FactorEngine,
#     AlphaModelTrainer
# )

# data_module = DataModule(data_config, strategy_config)
# data_module.load()

# factor_engine = FactorEngine(...)
# factor_engine.evaluate_expressions()

# alpha_trainer = AlphaModelTrainer(...)
# alpha_trainer.train_all_models()


# 运行完整流程
strategy.run_full_pipeline(
    weight_method='equal',       # 'equal' 或 'sharpe'
    normalize_method=None,       # None, 'simple', 'robust', 'zscore'
    enable_factor_selection=False
)

# 查看结果
strategy.plot_results('Ensemble')
strategy.plot_regime_and_risk('Ensemble')

# 诊断分析
strategy.diagnose_label_health()
df_ic = strategy.diagnose_factor_ic(data_range='train', top_n=10)
strategy.diagnose_top_factors(data_range='test', top_n=3)

# 保存模型
strategy.save_models('./saved_models')

plt.show()


# ========== 方式2: 从YAML + 因子表达式创建 ==========
"""
from multi_model_strategy import create_strategy_from_yaml

factor_expressions = [
    'ta_rsi_14(close)',
    'ta_ema_20(close)',
]

strategy = create_strategy_from_yaml(
    yaml_path='gp_crypto_next/coarse_grain_parameters.yaml',
    factor_expressions=factor_expressions,
    max_factors=10,
    enable_regime_layer=True,
    enable_risk_layer=True,
)

strategy.run_full_pipeline(weight_method='equal')
strategy.plot_results('Ensemble')
"""


# ========== 方式3: 从YAML + CSV文件创建（原有方式） ==========
"""
from multi_model_strategy import QuantTradingStrategy

strategy = QuantTradingStrategy.from_yaml(
    yaml_path='gp_crypto_next/coarse_grain_parameters.yaml',
    factor_csv_path='gp_models/ETHUSDT_15m_1_2025-01-01_2025-01-20.csv.gz',
    strategy_config={
        'max_factors': 30,
        'fees_rate': 0.0005,
    }
)

strategy.run_full_pipeline(weight_method='equal')
strategy.plot_results('Ensemble')
"""


# ========== 高级用法：使用 Triple Barrier + Kelly Bet Sizing ==========
"""
from multi_model_strategy import create_strategy_from_expressions

factors = ['ta_rsi_14(close)', 'ta_ema_20(close)']

strategy = create_strategy_from_expressions(
    factors,
    sym='ETHUSDT',
    train_dates=('2025-01-01', '2025-02-01'),
    test_dates=('2025-02-01', '2025-03-01'),
    # 启用 Triple Barrier
    use_triple_barrier_label=True,
    triple_barrier_pt_sl=[2, 2],
    triple_barrier_max_holding=[0, 4],
    # 启用 Kelly Bet Sizing
    use_kelly_bet_sizing=True,
    kelly_fraction=0.25,
)

strategy.run_full_pipeline()
strategy.plot_results('Ensemble')
"""


# ========== 模块化使用：手动控制每个步骤 ==========
"""
from multi_model_strategy import (
    QuantTradingStrategy,
    DataConfig,
    StrategyConfig
)

# 1. 创建策略实例
data_config = DataConfig.build_simple(
    sym='ETHUSDT',
    freq='15m',
    train_dates=('2025-01-01', '2025-02-01'),
    test_dates=('2025-02-01', '2025-03-01')
)

config = StrategyConfig.get_default_config()
config['max_factors'] = 10

strategy = QuantTradingStrategy(None, data_config, config)
strategy.factor_expressions = ['ta_rsi_14(close)', 'ta_ema_20(close)']
strategy._use_expressions_mode = True

# 2. 手动执行各步骤
strategy._load_data()
strategy._evaluate_factors(normalize_method=None, enable_factor_selection=False)
strategy._train_and_predict(weight_method='equal')
strategy._apply_position_scaling()
strategy._run_backtest()

# 3. 查看结果
strategy.plot_results('Ensemble')
"""

