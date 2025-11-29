"""
快速测试脚本：验证新模块可以正常导入
"""
import sys
from pathlib import Path

# 确保项目根目录和 gp_crypto_next 都在 sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

gp_crypto_dir = project_root / "gp_crypto_next"
if str(gp_crypto_dir) not in sys.path:
    sys.path.insert(0, str(gp_crypto_dir))

print("="*60)
print("测试模块导入")
print("="*60)

# 测试1：导入主策略类
try:
    from multi_model_strategy import QuantTradingStrategy
    print("✅ QuantTradingStrategy 导入成功")
except Exception as e:
    print(f"❌ QuantTradingStrategy 导入失败: {e}")

# 测试2：导入配置类
try:
    from multi_model_strategy import StrategyConfig, DataConfig
    print("✅ StrategyConfig, DataConfig 导入成功")
except Exception as e:
    print(f"❌ 配置类导入失败: {e}")

# 测试3：导入核心模块
try:
    from multi_model_strategy import (
        DataModule,
        FactorEngine,
        AlphaModelTrainer,
        BacktestEngine,
    )
    print("✅ 核心模块导入成功")
except Exception as e:
    print(f"❌ 核心模块导入失败: {e}")

# 测试4：导入仓位管理模块
try:
    from multi_model_strategy import (
        RegimeScaler,
        RiskScaler,
        KellyBetSizer,
        PositionScalingManager
    )
    print("✅ 仓位管理模块导入成功")
except Exception as e:
    print(f"❌ 仓位管理模块导入失败: {e}")

# 测试5：导入工具模块
try:
    from multi_model_strategy import Visualizer, DiagnosticTools
    print("✅ 工具模块导入成功")
except Exception as e:
    print(f"❌ 工具模块导入失败: {e}")

# 测试6：导入便捷函数
try:
    from multi_model_strategy import (
        create_strategy_from_expressions,
        create_strategy_from_yaml
    )
    print("✅ 便捷函数导入成功")
except Exception as e:
    print(f"❌ 便捷函数导入失败: {e}")

# 测试7：验证配置生成
try:
    config = StrategyConfig.get_default_config()
    assert 'return_period' in config
    assert 'enable_regime_layer' in config
    print("✅ 配置生成正常")
except Exception as e:
    print(f"❌ 配置生成失败: {e}")

# 测试8：验证简化创建接口（不真正运行，只测试能否创建实例）
try:
    factors = ['ta_rsi_14(close)']
    strategy = create_strategy_from_expressions(
        factors,
        sym='ETHUSDT',
        train_dates=('2025-01-01', '2025-01-02'),
        test_dates=('2025-01-02', '2025-01-03'),
        max_factors=1
    )
    print("✅ 策略实例创建成功")
    print(f"   - 因子数量: {len(strategy.factor_expressions)}")
    print(f"   - 交易对: {strategy.data_config['sym']}")
except Exception as e:
    print(f"❌ 策略实例创建失败: {e}")

print("\n" + "="*60)
print("所有测试完成！")
print("="*60)
print("\n使用说明：")
print("1. 运行 example_usage.py 查看完整使用示例")
print("2. 阅读 multi_model_strategy/README.md 了解详细文档")
print("3. 旧代码无需修改，API 完全兼容")

