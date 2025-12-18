import numpy as np
import pandas as pd
import talib as ta

# 创建测试数据
np.random.seed(42)
n = 20
x_array = np.random.randn(n)
y_array = np.random.randn(n)

# 转换为pandas Series
dates = pd.date_range('2023-01-01', periods=n)
x_series = pd.Series(x_array, index=dates)
y_series = pd.Series(y_array, index=dates)

print("测试数据:")
print(f"x前5个值: {x_array[:5]}")
print(f"y前5个值: {y_array[:5]}")
print()

# 测试方法1 - talib.CORREL
print("测试方法1 - ta.CORREL:")
try:
    result1 = ta.CORREL(x_array, y_array, 10)
    print(f"结果: {result1}")
except Exception as e:
    print(f"错误: {e}")
print()

# 测试方法2 - pandas rolling
print("测试方法2 - pandas rolling:")
try:
    result2 = x_series.rolling(10).corr(y_series)
    print(f"结果前几个值:")
    print(result2.head(15))
except Exception as e:
    print(f"错误: {e}")
print()

# 手动验证一个窗口的相关系数
print("手动验证第10个位置的相关系数:")
window_x = x_array[0:10]  # 前10个x值
window_y = y_array[0:10]  # 前10个y值
manual_corr = np.corrcoef(window_x, window_y)[0,1]
print(f"手动计算的相关系数: {manual_corr}")
print(f"pandas计算的相关系数: {result2.iloc[9]}") 