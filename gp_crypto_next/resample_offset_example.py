"""
使用offset参数进行降频的示例代码

演示新旧两种方法的对比，以及如何使用 data_prepare_coarse_grain_rolling_offset
"""

import pandas as pd
import numpy as np

# ===== 示例1: 对比两种resample方法 =====

print("="*60)
print("示例1: 对比时间偏移 vs offset参数")
print("="*60)

# 创建示例数据：从9:15开始的15分钟数据
timestamps = pd.date_range('2024-01-01 09:15', periods=10, freq='15min')
df = pd.DataFrame({
    'o': np.random.randn(10) + 100,
    'h': np.random.randn(10) + 101,
    'l': np.random.randn(10) + 99,
    'c': np.random.randn(10) + 100,
    'vol': np.random.randint(100, 1000, 10),
    'vol_ccy': np.random.randint(10000, 100000, 10),
    'trades': np.random.randint(50, 500, 10),
}, index=timestamps)

print("\n原始数据（15分钟）:")
print(df.index)

# 方法1: 默认resample（整点对齐）
result_default = df.resample('1H', closed='left', label='left').agg({
    'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last',
    'vol': 'sum', 'vol_ccy': 'sum', 'trades': 'sum'
})
print("\n默认resample（对齐到整点）:")
print(result_default.index)
# 输出: [09:00, 10:00, 11:00] - 对齐到整点

# 方法2: 使用offset参数（推荐✨）
result_offset = df.resample('1H', closed='left', label='left', 
                            offset=pd.Timedelta(minutes=15)).agg({
    'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last',
    'vol': 'sum', 'vol_ccy': 'sum', 'trades': 'sum'
})
print("\n使用offset参数（对齐到9:15）:")
print(result_offset.index)
# 输出: [09:15, 10:15, 11:15] - 对齐到:15分

# 方法3: 时间偏移（不推荐❌，容易出错）
df_shifted = df.copy()
df_shifted.index = df_shifted.index - pd.Timedelta(minutes=15)
result_shifted = df_shifted.resample('1H', closed='left', label='left').agg({
    'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last',
    'vol': 'sum', 'vol_ccy': 'sum', 'trades': 'sum'
})
result_shifted.index = result_shifted.index + pd.Timedelta(minutes=15)
print("\n使用时间偏移:")
print(result_shifted.index)

# 验证方法2和方法3的结果是否一致
print("\n验证: offset参数 vs 时间偏移结果是否相同?")
print(f"索引相同: {result_offset.index.equals(result_shifted.index)}")
print(f"数据相同: {result_offset.equals(result_shifted)}")


# ===== 示例2: 在实际项目中使用新方法 =====

print("\n" + "="*60)
print("示例2: 使用 data_prepare_coarse_grain_rolling_offset")
print("="*60)

# 示例代码（假设已经导入了dataload模块）
example_code = """
from dataload import data_prepare_coarse_grain_rolling_offset

# 使用新的offset参数版本
X_all, X_train, y_train, ret_train, X_test, y_test, ret_test, \\
    feature_names, open_train, open_test, close_train, close_test, \\
    timestamps, ohlc_aligned, y_p_train, y_p_test = \\
    data_prepare_coarse_grain_rolling_offset(
        sym='BTCUSDT',
        freq='2h',
        start_date_train='2021-12-01',
        end_date_train='2024-01-01',
        start_date_test='2024-01-01',
        end_date_test='2024-09-10',
        coarse_grain_period='1h',      # 粗粒度周期
        feature_lookback_bars=8,       # 特征窗口: 8个1小时桶
        rolling_step='15min',          # 滚动步长: 15分钟
        y_train_ret_period=8,          # 预测周期: 8个15分钟 = 2小时
        rolling_w=2000,
        use_fine_grain_precompute=True,
        timeframe='15m',
        file_path='path/to/data.csv.gz'
    )

# 关键区别：
# 1. 使用offset参数，避免时间索引偏移
# 2. 代码更简洁，逻辑更清晰
# 3. 减少边界问题和潜在错误
"""

print(example_code)


# ===== 示例3: 多组offset的效果演示 =====

print("\n" + "="*60)
print("示例3: 多组offset覆盖所有时间点")
print("="*60)

# 假设粗粒度周期是1小时，滚动步长是15分钟
# 需要4组不同offset的桶来覆盖所有可能的起始点

offsets = [
    pd.Timedelta(minutes=0),   # [9:00, 10:00, 11:00]
    pd.Timedelta(minutes=15),  # [9:15, 10:15, 11:15]
    pd.Timedelta(minutes=30),  # [9:30, 10:30, 11:30]
    pd.Timedelta(minutes=45),  # [9:45, 10:45, 11:45]
]

print("\n原始时间点（15分钟）:")
print(df.index.strftime('%H:%M').tolist())

print("\n不同offset的降频结果:")
for i, offset in enumerate(offsets):
    result = df.resample('1H', closed='right', label='right', offset=offset).agg({
        'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last',
        'vol': 'sum'
    })
    print(f"组{i} (offset={offset}): {result.index.strftime('%H:%M').tolist()}")

print("\n✅ 这样确保每个15分钟的时间点都有对应的1小时粗粒度特征")


# ===== 关键优势总结 =====

print("\n" + "="*60)
print("使用offset参数的关键优势")
print("="*60)

advantages = """
1. 🔒 **更安全**: 
   - 不修改原始时间索引，避免边界问题
   - pandas原生参数，经过充分测试
   
2. 🎯 **更准确**:
   - 时间对齐逻辑清晰，不易出错
   - 避免了"偏移-处理-恢复"三步操作可能的精度损失
   
3. 📖 **更易读**:
   - 代码意图明确，一眼看懂
   - 减少代码行数，降低维护成本
   
4. ⚡ **性能相当**:
   - pandas内部优化，性能不输时间偏移方法
   - 减少了copy操作，可能更快

5. 🐛 **更易调试**:
   - 不需要担心偏移恢复是否正确
   - 时间戳边界问题更容易发现和修复
"""

print(advantages)

print("\n" + "="*60)
print("使用建议")
print("="*60)

recommendations = """
- ✅ 新项目：优先使用 data_prepare_coarse_grain_rolling_offset
- ✅ 重构代码：逐步迁移到offset参数版本
- ⚠️  旧项目：如果当前方法运行稳定，可以保持不变
- 📝 测试：迁移后务必对比新旧方法的输出结果
"""

print(recommendations)

