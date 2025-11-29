# 迁移指南：从 multi_model_main.py 到模块化结构

---

## 📦 新模块结构总览

```
multi_model_strategy/              # 新模块包
├── __init__.py                    # 包入口（导出所有主要类）
├── config.py                      # 配置管理
├── data_module.py                 # 数据加载
├── factor_engine.py               # 因子引擎
├── alpha_models.py                # 模型训练
├── position_scaling.py            # 仓位缩放（Regime + Risk + Kelly）
├── backtest_engine.py             # 回测引擎
├── visualization.py               # 可视化
├── diagnostics.py                 # 诊断工具
├── strategy.py                    # 主策略类（整合层）
└── README.md                      # 详细文档

example_usage.py                   # 使用示例
test_import.py                     # 测试脚本
MIGRATION_GUIDE.md                 # 本文档
```

---

## ✅ 向后兼容性

**好消息：旧代码无需修改！**

### 旧代码（仍然有效）：
```python
# 原来的导入方式仍然可用
from multi_model_main import QuantTradingStrategy

strategy = QuantTradingStrategy.from_yaml(...)
strategy.run_full_pipeline()
strategy.plot_results('Ensemble')
```

### 新代码（推荐）：
```python
# 新的导入方式（更清晰）
from multi_model_strategy import QuantTradingStrategy

strategy = QuantTradingStrategy.from_yaml(...)
strategy.run_full_pipeline()  # API 完全一致！
```

---

## 🚀 迁移建议（循序渐进）

### 阶段 1：无缝迁移（零风险）
**时间：立即**

只需改变导入语句：
```python
# 旧
from multi_model_main import QuantTradingStrategy

# 新
from multi_model_strategy import QuantTradingStrategy
```

其余代码**完全不变**。

---

### 阶段 2：使用便捷函数（提升效率）
**时间：在新项目中**

对于快速原型和测试，使用新的便捷函数：

```python
from multi_model_strategy import create_strategy_from_expressions

# 无需 YAML，直接创建
strategy = create_strategy_from_expressions(
    factor_expressions=['ta_rsi_14(close)', 'ta_ema_20(close)'],
    sym='ETHUSDT',
    train_dates=('2025-01-01', '2025-02-01'),
    test_dates=('2025-02-01', '2025-03-01'),
    max_factors=5,
    fees_rate=0.0005
)

strategy.run_full_pipeline()
```

**优势**：
- 无需创建 YAML 配置文件
- 代码更简洁
- 适合快速测试

---

### 阶段 3：模块化使用（高级控制）
**时间：需要精细控制时**

对于复杂场景，直接使用各个子模块：

```python
from multi_model_strategy import (
    DataModule,
    FactorEngine,
    AlphaModelTrainer,
    PositionScalingManager,
    BacktestEngine
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

# 3. 训练模型
alpha_trainer = AlphaModelTrainer(...)
alpha_trainer.train_all_models()

# ... 精细控制每个步骤
```

**优势**：
- 每个模块可独立测试
- 易于调试
- 可插拔式设计

---

## 🔄 功能对照表

| 旧代码（multi_model_main.py）| 新模块 | 说明 |
|---|---|---|
| `QuantTradingStrategy.__init__` | `strategy.py` | 主策略类（保持兼容）|
| `load_data_from_dataload()` | `data_module.py` | 数据加载逻辑 |
| `evaluate_factor_expressions()` | `factor_engine.py` | 因子评估 |
| `normalize_factors()` | `factor_engine.py` | 因子标准化 |
| `select_factors()` | `factor_engine.py` | 因子筛选 |
| `train_models()` | `alpha_models.py` | 模型训练 |
| `make_predictions()` | `alpha_models.py` | 模型预测与集成 |
| `build_regime_scaler()` | `position_scaling.py::RegimeScaler` | Regime 层 |
| `build_risk_scaler()` | `position_scaling.py::RiskScaler` | Risk 层 |
| `apply_kelly_bet_sizing()` | `position_scaling.py::KellyBetSizer` | Kelly sizing |
| `real_trading_simulator()` | `backtest_engine.py` | 回测模拟器 |
| `plot_results()` | `visualization.py::Visualizer` | 回测可视化 |
| `plot_regime_and_risk_scalers()` | `visualization.py::Visualizer` | Regime/Risk 诊断 |
| `diagnose_label_health()` | `diagnostics.py::DiagnosticTools` | Label 诊断 |
| `diagnose_factor_ic()` | `diagnostics.py::DiagnosticTools` | IC 计算 |
| `backtest_single_factor()` | `diagnostics.py::DiagnosticTools` | 单因子回测 |

---

## 📝 配置文件变化

### 配置不变！

所有 YAML 配置文件格式**完全不变**，可直接使用：

```python
# 旧代码
strategy = QuantTradingStrategy.from_yaml('config.yaml', 'factors.csv.gz')

# 新代码（完全兼容）
from multi_model_strategy import QuantTradingStrategy
strategy = QuantTradingStrategy.from_yaml('config.yaml', 'factors.csv.gz')
```

---

## 🎯 使用场景推荐

### 场景 1：生产环境（稳定性优先）
**保持旧代码不变**，只改导入语句。

```python
from multi_model_strategy import QuantTradingStrategy
# 其余代码完全不变
```

---

### 场景 2：快速测试（效率优先）
**使用便捷函数**。

```python
from multi_model_strategy import create_strategy_from_expressions

strategy = create_strategy_from_expressions(
    factors=['ta_rsi_14(close)'],
    sym='ETHUSDT',
    train_dates=('2025-01-01', '2025-02-01'),
    test_dates=('2025-02-01', '2025-03-01')
)
strategy.run_full_pipeline()
```

---

### 场景 3：研究开发（灵活性优先）
**直接使用子模块**。

```python
from multi_model_strategy import (
    FactorEngine,
    AlphaModelTrainer,
    DiagnosticTools
)

# 精细控制每个步骤
factor_engine = FactorEngine(...)
factor_engine.evaluate_expressions()
factor_engine.normalize(method='robust')

# 单独测试某个模块
```

---

## 🐛 常见问题

### Q1: 旧代码还能用吗？
**A:** 能！`multi_model_main.py` 保持不变，所有旧代码无需修改。

### Q2: 导入报错怎么办？
**A:** 确保项目根目录在 `sys.path` 中：
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

### Q3: 性能有变化吗？
**A:** 无变化。新模块只是重新组织代码，核心逻辑完全一致。

### Q4: 需要安装新依赖吗？
**A:** 不需要。依赖项完全一致。

### Q5: 如何运行测试？
**A:** 
```bash
# 测试导入
python test_import.py

# 查看使用示例
python example_usage.py
```

---

## 📚 学习路径

1. **第1天**：阅读 `multi_model_strategy/README.md`，了解模块结构
2. **第2天**：运行 `example_usage.py`，熟悉新 API
3. **第3天**：在新项目中使用 `create_strategy_from_expressions`
4. **第4天**：尝试模块化使用，单独调用 `FactorEngine`、`AlphaModelTrainer`
5. **第5天**：为自己的策略定制新的缩放层或诊断工具

---

## ✨ 新功能亮点

相比旧代码，新模块提供：

1. **更清晰的结构**：8 个独立模块，各司其职
2. **便捷函数**：`create_strategy_from_expressions` 无需 YAML
3. **可测试性**：每个模块可独立测试
4. **可扩展性**：易于添加新模型、新缩放层
5. **完整文档**：每个模块都有 docstring 和 README

---

## 🎉 总结

**核心原则：渐进式迁移，零风险**

- ✅ 旧代码无需修改（只改导入）
- ✅ 新功能向后兼容
- ✅ 模块化设计，易于扩展
- ✅ 完整文档，快速上手

**立即行动：**
1. 运行 `python test_import.py` 验证导入
2. 查看 `example_usage.py` 学习新 API
3. 在新项目中尝试便捷函数

**问题反馈：**
如遇到任何问题，请检查 `multi_model_strategy/README.md` 或提 issue。

---

🎯 **Enjoy the new modular structure!**

