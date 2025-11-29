# Multi-Model Quantitative Trading Strategy

多模型量化交易策略框架（整合 GP 因子）

---

## 📁 模块结构

```
multi_model_strategy/
├── __init__.py              # 包入口，导出主要类和便捷函数
├── config.py                # 配置管理（DataConfig, StrategyConfig）
├── data_module.py           # 数据加载（封装 dataload）
├── factor_engine.py         # 因子评估、标准化、筛选
├── alpha_models.py          # 模型训练与预测（OLS/Ridge/XGB/LGB）
├── position_scaling.py      # 仓位缩放（Regime + Risk + Kelly）
├── backtest_engine.py       # 回测模拟器
├── visualization.py         # 可视化（回测结果、Regime/Risk诊断）
├── diagnostics.py           # 诊断工具（Label健康度、IC、单因子回测）
└── strategy.py              # 主策略类（整合所有模块）
```

---

## ✨ 核心功能

### 1. **因子引擎**（`factor_engine.py`）
- 评估 GP 因子表达式
- 3 种标准化方法：`robust`、`zscore`、`simple`
- 基于相关性的因子筛选

### 2. **模型训练**（`alpha_models.py`）
- 支持 5 种模型：`LinearRegression`, `Ridge`, `Lasso`, `XGBoost`, `LightGBM`
- 等权重 / 基于 Sharpe 的模型集成

### 3. **仓位缩放**（`position_scaling.py`）
- **Regime 层**：基于趋势强度和波动水平调整仓位
- **Risk 层**：基于拥挤度、冲击、资金成本调整仓位
- **Kelly Bet Sizing**：Lopez 风格（胜率 × 盈亏比）

### 4. **回测引擎**（`backtest_engine.py`）
- 真实交易模拟（滑点、手续费）
- 性能指标：年化收益、Sharpe、最大回撤、Calmar、胜率、盈亏比

### 5. **诊断工具**（`diagnostics.py`）
- Label 健康度检查
- 因子 IC / RankIC 计算
- 单因子多空回测

---

## 🚀 快速开始

### 方式 1：最简单的方式（推荐）

```python
from multi_model_strategy import create_strategy_from_expressions

# 定义因子
factors = [
    'ta_rsi_14(close)',
    'ta_ema_20(close)',
]

# 创建策略
strategy = create_strategy_from_expressions(
    factors,
    sym='ETHUSDT',
    train_dates=('2025-01-01', '2025-02-01'),
    test_dates=('2025-02-01', '2025-03-01'),
    max_factors=5
)

# 运行
strategy.run_full_pipeline()
strategy.plot_results('Ensemble')
```

### 方式 2：从 YAML 配置创建

```python
from multi_model_strategy import create_strategy_from_yaml

factors = ['ta_rsi_14(close)', 'ta_ema_20(close)']

strategy = create_strategy_from_yaml(
    'config.yaml',
    factors,
    max_factors=10
)

strategy.run_full_pipeline()
strategy.plot_results('Ensemble')
```

### 方式 3：从 CSV 文件加载因子

```python
from multi_model_strategy import QuantTradingStrategy

strategy = QuantTradingStrategy.from_yaml(
    yaml_path='config.yaml',
    factor_csv_path='factors.csv.gz',
    strategy_config={'max_factors': 30}
)

strategy.run_full_pipeline()
```

---

## 🔧 高级功能

### 启用 Triple Barrier + Kelly Bet Sizing

```python
strategy = create_strategy_from_expressions(
    factors,
    sym='ETHUSDT',
    train_dates=('2025-01-01', '2025-02-01'),
    test_dates=('2025-02-01', '2025-03-01'),
    # Triple Barrier
    use_triple_barrier_label=True,
    triple_barrier_pt_sl=[2, 2],
    triple_barrier_max_holding=[0, 4],
    # Kelly Bet Sizing
    use_kelly_bet_sizing=True,
    kelly_fraction=0.25,
)

strategy.run_full_pipeline()
```

### 诊断与分析

```python
# Label 健康度
strategy.diagnose_label_health()

# 因子 IC 分析
df_ic = strategy.diagnose_factor_ic(data_range='train', top_n=20)

# Top 因子回测
strategy.diagnose_top_factors(data_range='test', top_n=5)

# Regime & Risk 可视化
strategy.plot_regime_and_risk('Ensemble')
```

### 保存模型

```python
strategy.save_models('./saved_models')
```

---

## 📊 配置说明

### 数据配置（`DataConfig`）

```python
data_config = {
    'sym': 'ETHUSDT',
    'freq': '15m',
    'start_date_train': '2025-01-01',
    'end_date_train': '2025-02-01',
    'start_date_test': '2025-02-01',
    'end_date_test': '2025-03-01',
    'rolling_window': 2000,
    'data_source': 'coarse_grain',  # 'kline' 或 'coarse_grain'
    'coarse_grain_period': '2h',
    'feature_lookback_bars': 8,
}
```

### 策略配置（`StrategyConfig`）

```python
strategy_config = {
    'return_period': 1,
    'corr_threshold': 0.5,
    'fees_rate': 0.0005,
    'max_factors': 10,
    'clip_num': 5.0,
    'annual_bars': 35040,  # 15分钟 K线
    
    # 三层结构开关
    'enable_regime_layer': True,
    'enable_risk_layer': True,
    
    # Triple Barrier
    'use_triple_barrier_label': False,
    'triple_barrier_pt_sl': [2, 2],
    'triple_barrier_max_holding': [0, 4],
    
    # Kelly Bet Sizing
    'use_kelly_bet_sizing': False,
    'kelly_fraction': 0.25,
}
```

---

## 🎯 模块化使用（手动控制）

```python
from multi_model_strategy import (
    DataModule,
    FactorEngine,
    AlphaModelTrainer,
    PositionScalingManager,
    BacktestEngine,
)

# 1. 加载数据
data_module = DataModule(data_config, strategy_config)
data_module.load()

# 2. 评估因子
factor_engine = FactorEngine(
    factor_expressions, 
    data_module.X_all, 
    data_module.feature_names,
    data_module.y_train
)
factor_engine.evaluate_expressions()
factor_engine.normalize(method='robust')
factor_engine.select_by_correlation(corr_threshold=0.5)

# 3. 训练模型
alpha_trainer = AlphaModelTrainer(X_train, X_test, y_train, y_test, selected_factors)
alpha_trainer.train_all_models()
alpha_trainer.make_predictions()
alpha_trainer.ensemble_models(weight_method='equal')

# 4. 仓位缩放
position_manager = PositionScalingManager(config, feature_df, train_len)
position_manager.build_regime_and_risk_scalers()
predictions = position_manager.apply_to_predictions(predictions)

# 5. 回测
backtest_engine = BacktestEngine(open_train, close_train, open_test, close_test, fees_rate)
results = backtest_engine.backtest_all_models(predictions)
```

---

## 📝 关键设计原则

1. **单一职责**：每个模块只负责一个明确的功能域
2. **低耦合**：模块间通过接口交互，减少直接依赖
3. **向后兼容**：保持原有 `QuantTradingStrategy` API 不变
4. **可测试性**：每个模块可独立测试
5. **可扩展性**：易于添加新模型、新缩放层、新诊断工具

---

## 🔄 迁移指南（从旧版本）

### 旧代码：
```python
from multi_model_main import QuantTradingStrategy

strategy = QuantTradingStrategy(...)
strategy.run_full_pipeline()
```

### 新代码（完全兼容）：
```python
from multi_model_strategy import QuantTradingStrategy

strategy = QuantTradingStrategy(...)
strategy.run_full_pipeline()  # API 不变！
```

或使用新的便捷函数：
```python
from multi_model_strategy import create_strategy_from_expressions

strategy = create_strategy_from_expressions(factors, ...)
strategy.run_full_pipeline()
```

---

## 🛠️ 依赖项

- Python 3.7+
- NumPy, Pandas
- scikit-learn
- XGBoost, LightGBM
- matplotlib
- gp_crypto_next（项目内部模块）

---

## 📚 更多示例

参见 `example_usage.py` 文件，包含完整的使用示例。

---

## 🐛 问题排查

### 1. 导入错误
确保项目根目录在 `sys.path` 中：
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

### 2. 数据加载失败
检查 `data_config` 中的路径和日期范围是否正确。

### 3. 因子评估失败
确保因子表达式语法正确，且所需的基础特征存在于 `feature_names` 中。

---

## 📄 License

MIT License

