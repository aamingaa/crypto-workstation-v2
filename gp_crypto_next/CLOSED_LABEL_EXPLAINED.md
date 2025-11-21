# Resample 的 closed 和 label 参数详解

## 关键问题：原始数据的时间戳含义是什么？

在使用 `resample()` 前，**必须先确认原始数据的时间戳表示什么**。

---

## 两种常见的时间戳约定

### 1. 时间戳 = K线的**开始时间**（Open Time）

**示例：** Binance、OKX 等大部分交易所

```
时间戳: 09:00  →  表示 [09:00, 09:01) 的K线
时间戳: 09:01  →  表示 [09:01, 09:02) 的K线
时间戳: 09:02  →  表示 [09:02, 09:03) 的K线
```

**应该使用：** `closed='left', label='left'`

```python
# 正确做法
df.resample('1H', closed='left', label='left').agg({
    'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'vol': 'sum'
})
```

**含义：**
- `closed='left'`: 区间左闭右开 [09:00, 10:00)
  - 包含 09:00, 09:01, ..., 09:59
  - 不包含 10:00
- `label='left'`: 用左端点作为标签 → 09:00

**结果：**
```
原始数据: [09:00, 09:01, 09:02, ..., 09:59, 10:00, 10:01, ...]
聚合结果: 
  09:00 → 包含 [09:00, 09:01, ..., 09:59]
  10:00 → 包含 [10:00, 10:01, ..., 10:59]
```

---

### 2. 时间戳 = K线的**结束时间**（Close Time）

**示例：** 部分传统金融数据源

```
时间戳: 09:01  →  表示 (09:00, 09:01] 的K线
时间戳: 09:02  →  表示 (09:01, 09:02] 的K线
时间戳: 09:03  →  表示 (09:02, 09:03] 的K线
```

**应该使用：** `closed='right', label='right'`

```python
# 正确做法
df.resample('1H', closed='right', label='right').agg({
    'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'vol': 'sum'
})
```

**含义：**
- `closed='right'`: 区间左开右闭 (09:00, 10:00]
  - 不包含 09:00
  - 包含 09:01, ..., 10:00
- `label='right'`: 用右端点作为标签 → 10:00

---

## 如何判断您的数据？

### 方法1: 查看数据源文档

**Binance 官方文档明确说明：**
```
[
  1499040000000,      // 开盘时间 (Open time)
  "0.01634790",       // 开盘价
  "0.80000000",       // 最高价
  "0.01575800",       // 最低价
  "0.01577100",       // 收盘价
  "148976.11427815",  // 成交量
  1499644799999,      // 收盘时间 (Close time)
  "2434.19055334",    // ...
]
```

**关键：** 第一列（index=0）是 **Open time**，时间戳表示K线开始时间

### 方法2: 实际验证

```python
import pandas as pd

# 读取您的数据
df = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)
df = df.sort_index()

# 查看前几行
print(df.head())
print(f"\n时间间隔: {df.index[1] - df.index[0]}")

# 假设是1分钟数据，检查时间戳的规律
# 如果时间戳都是整分钟（09:00, 09:01, 09:02）→ Open Time
# 如果时间戳都是整分钟+1秒（09:00:59, 09:01:59）→ Close Time
```

---

## 您的代码分析

### 当前代码（Line 794）

```python
coarse_bars = resample(z_raw_offset, coarse_grain_period, closed='right', label='right')
```

**问题分析：**

如果您的原始数据来自 **Binance**：
- ❌ **错误**：应该使用 `closed='left', label='left'`
- 当前使用 `closed='right', label='right'` 会导致：
  - 区间划分不正确
  - 每个K线都"晚了一个单位"

### 正确的做法

```python
# 如果数据来自 Binance（Open Time）
coarse_bars = resample(z_raw_offset, coarse_grain_period, closed='left', label='left')
```

---

## 实际影响示例

### 场景：15分钟数据 → 1小时聚合

**原始数据（时间戳=Open Time）：**
```
09:00, 09:15, 09:30, 09:45, 10:00, 10:15, 10:30, 10:45, 11:00
```

#### ❌ 错误做法：`closed='right', label='right'`

```python
result = df.resample('1H', closed='right', label='right').agg(...)
```

**结果：**
```
10:00 → 包含 (09:00, 10:00] = [09:15, 09:30, 09:45, 10:00]
11:00 → 包含 (10:00, 11:00] = [10:15, 10:30, 10:45, 11:00]
```

**问题：**
- 09:00 的数据**被丢弃**了！
- 每个小时桶的第一根K线（整点）被归到了上一个小时

#### ✅ 正确做法：`closed='left', label='left'`

```python
result = df.resample('1H', closed='left', label='left').agg(...)
```

**结果：**
```
09:00 → 包含 [09:00, 10:00) = [09:00, 09:15, 09:30, 09:45]
10:00 → 包含 [10:00, 11:00) = [10:00, 10:15, 10:30, 10:45]
11:00 → 包含 [11:00, 12:00) = [11:00, ...]
```

**正确：**
- 没有数据丢失
- 每个小时桶从整点开始，符合直觉

---

## 推荐修改

### 修改1: dataload.py Line 794

```python
# 旧代码
coarse_bars = resample(z_raw_offset, coarse_grain_period, closed='right', label='right')

# 新代码（如果数据是Binance Open Time格式）
coarse_bars = resample(z_raw_offset, coarse_grain_period, closed='left', label='left')
```

### 修改2: dataload.py Line 1089-1090（新方法）

```python
# 旧代码
coarse_bars = resample_with_offset(
    z_raw.copy(), 
    coarse_grain_period, 
    offset=offset,
    closed='right', 
    label='right'
)

# 新代码（如果数据是Binance Open Time格式）
coarse_bars = resample_with_offset(
    z_raw.copy(), 
    coarse_grain_period, 
    offset=offset,
    closed='left', 
    label='left'
)
```

### 修改3: resample函数的默认值

```python
# 当前默认值
def resample(z: pd.DataFrame, freq: str, closed: str = 'left', label: str = 'left') -> pd.DataFrame:
```

**好消息：** 默认值已经是 `'left'`，这是正确的！

**问题：** 调用时**显式传入了** `closed='right', label='right'`，覆盖了正确的默认值。

---

## 验证方法

### 测试脚本

```python
import pandas as pd
import numpy as np

# 创建测试数据（模拟Binance格式：时间戳=Open Time）
timestamps = pd.date_range('2024-01-01 09:00', periods=8, freq='15min')
df = pd.DataFrame({
    'o': [100, 101, 102, 103, 104, 105, 106, 107],
    'h': [101, 102, 103, 104, 105, 106, 107, 108],
    'l': [99, 100, 101, 102, 103, 104, 105, 106],
    'c': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5],
    'vol': [1000]*8,
}, index=timestamps)

print("原始数据（15分钟）:")
print(df)
print()

# 错误做法
result_wrong = df.resample('1H', closed='right', label='right').agg({
    'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'vol': 'sum'
})
print("错误做法 (closed='right', label='right'):")
print(result_wrong)
print()

# 正确做法
result_correct = df.resample('1H', closed='left', label='left').agg({
    'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'vol': 'sum'
})
print("正确做法 (closed='left', label='left'):")
print(result_correct)
print()

# 对比
print("数据量对比:")
print(f"  错误做法: {len(result_wrong)} 行")
print(f"  正确做法: {len(result_correct)} 行")
```

**预期输出：**
```
错误做法: 2行（10:00, 11:00）- 丢失了09:00的桶
正确做法: 3行（09:00, 10:00, 11:00）- 完整保留
```

---

## 总结与建议

### 1. 确认数据源

| 数据源 | 时间戳含义 | 应使用参数 |
|--------|----------|-----------|
| **Binance** | Open Time | `closed='left', label='left'` ✅ |
| **OKX** | Open Time | `closed='left', label='left'` ✅ |
| **某些传统金融数据** | Close Time | `closed='right', label='right'` |

### 2. 修改建议

**如果您的数据来自加密货币交易所（99%概率）：**

```python
# 全局搜索并替换
# 旧: closed='right', label='right'
# 新: closed='left', label='left'
```

### 3. 测试验证

修改后，务必验证：
```python
# 1. 检查数据量
print(f"原始数据: {len(z_raw)} 行")
print(f"聚合后: {len(coarse_bars)} 行")
print(f"预期: {len(z_raw) / (粗粒度 / 细粒度)} 行")

# 2. 检查时间范围
print(f"原始: {z_raw.index.min()} ~ {z_raw.index.max()}")
print(f"聚合: {coarse_bars.index.min()} ~ {coarse_bars.index.max()}")

# 3. 检查第一个和最后一个桶是否正确
print(coarse_bars.head())
print(coarse_bars.tail())
```

### 4. 为什么之前没发现问题？

可能的原因：
- 数据量很大，丢失一点边界数据不明显
- offset操作掩盖了问题
- 模型训练没有完全使用边界数据

但**正确的参数仍然很重要**，特别是在：
- 回测时需要精确时间对齐
- 计算IC等统计指标
- 边界数据很关键的场景

---

## 快速检查清单

- [ ] 确认数据源的时间戳含义
- [ ] 修改 Line 794 的 closed/label 参数
- [ ] 修改 Line 1089-1090 的 closed/label 参数
- [ ] 运行测试脚本验证
- [ ] 对比修改前后的结果
- [ ] 更新相关文档

需要帮助实施这些修改吗？

