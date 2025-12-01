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
    # 'ta_dema_55(ta_dx_25(neg(h), ta_trix_8(h), ta_trix_21(ori_trix_21)))',
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


best = strategy.run_param_search(
    signal_grid=[60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0],         # 只搜 3 个阈值
    pt_sl_grid=[[2.0, 2.0], [3.0, 2.0]],               # 固定 pt_sl，不搜索
    max_holding_grid=[[0, 8], [0, 10], [0,12], [0,14], [0, 16]],             # 固定 max_holding
    metric='Sharpe Ratio',                 # 按 Calmar 选
    data_range='test',                     # 用 test 段评估
    model_name='Ensemble',                 # 看 Ensemble 的表现
    weight_method='equal',
    normalize_method=None,
    enable_factor_selection=False,
)

print(best)

# 运行完整流程
# strategy.run_full_pipeline(
#     weight_method='equal',       # 'equal' 或 'sharpe'
#     normalize_method=None,       # None, 'simple', 'robust', 'zscore'
#     enable_factor_selection=False
# )

# 查看结果并保存图像
# strategy.plot_results('Ensemble')
# # plt.savefig('ensemble_backtest.png', dpi=200, bbox_inches='tight')
# start_time = '2025-02-01 00:00:00'
# end_time = '2025-02-10 00:00:00'

# pnl_sub, metrics_sub = strategy.backtest_subperiod_by_time(
#     start_time=start_time,
#     end_time=end_time,
#     model_name='Ensemble',   # 或其它模型名
#     data_range='test',       # 'train' 或 'test'
# )


# strategy.plot_regime_and_risk('Ensemble')




# 诊断分析
# strategy.diagnose_label_health()
# df_ic = strategy.diagnose_factor_ic(data_range='train', top_n=10)
# strategy.diagnose_top_factors(data_range='test', top_n=3)

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