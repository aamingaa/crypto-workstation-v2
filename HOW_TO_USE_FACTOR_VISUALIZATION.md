# 单因子可视化使用指南

## 功能说明

`visualize_factor_vs_price` 方法用于在同一张图上叠加显示价格曲线和因子曲线（双 Y 轴），便于观察因子与价格的时序关系。

**典型场景**：观察价格见顶下跌前，Liq_Zscore（清算压力因子）是否出现尖峰。

---

## 快速使用

```python
# 假设你已经有了 diag 对象（DiagnosticTools 实例）

# 可视化某个因子
diag.visualize_factor_vs_price(
    factor_name="gp_factor_0",  # 替换为你的因子名称
    data_range='test',          # 'train' 或 'test'
    price_type='close',         # 'close' 或 'open'
    save_dir="./diagnostics",   # 保存路径
    figsize=(14, 8)             # 图形大小
)
```

---

## 完整示例

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "gp_crypto_next"))
from main_gp_new import MainGPTrainer

# 1. 初始化训练器
trainer = MainGPTrainer(
    sym='ETHUSDT',
    freq='15m',
    start_date_train='2022-01-01',
    end_date_train='2025-01-01',
    start_date_test='2025-01-01',
    end_date_test='2025-10-01',
    y_train_ret_period='2h',
    rolling_window=8,
    data_source='coarse_grain',
)

# 2. 读取因子池
exp_pool = trainer.read_and_pick()

# 3. 构建诊断工具
diag = trainer.build_diagnostic_tools_from_exp_pool(
    exp_pool=exp_pool,
    fees_rate=0.0005
)

# 4. 查看可用因子
print("可用因子列表：")
print(diag.selected_factors[:10])

# 5. 可视化因子与价格
diag.visualize_factor_vs_price(
    factor_name="gp_factor_0",  # 替换为实际因子名
    data_range='test',
    price_type='close',
    save_dir=trainer.total_factor_file_dir / "diagnostics"
)
```

---

## 批量可视化 Top N 因子

```python
# 先计算 IC，找出表现最好的因子
df_ic = diag.diagnose_factor_ic(data_range='test', top_n=10)

if df_ic is not None:
    # 取 Top 5
    top_5 = df_ic['factor'].head(5).tolist()
    
    # 批量可视化
    for factor_name in top_5:
        print(f"可视化因子: {factor_name}")
        diag.visualize_factor_vs_price(
            factor_name=factor_name,
            data_range='test',
            price_type='close',
            save_dir="./diagnostics/top_factors"
        )
```

---

## 参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `factor_name` | str | 因子名称（必须存在于 factor_data 中） | - |
| `data_range` | str | 'train' 或 'test' | 'test' |
| `price_type` | str | 'close' 或 'open' | 'close' |
| `save_dir` | str/Path | 保存路径，None 则直接显示 | None |
| `figsize` | tuple | 图形尺寸 (宽, 高) | (14, 8) |

---

## 输出图表说明

生成的图表包含：

- **蓝色线（左 Y 轴）**：价格曲线
- **红色线（右 Y 轴）**：因子曲线
- **灰色虚线**：因子均值
- **橙色虚线**：因子 ±1σ 边界

---

## 如何观察因子有效性

### 1. 价格与因子的同步性
- 价格上涨时，因子是上升还是下降？
- 因子变化是领先、同步还是滞后于价格？

### 2. 拐点信号
- 价格见顶前，因子是否出现尖峰？
- 价格见底前，因子是否出现谷底？

### 3. 信号稳定性
- 因子波动是否在 ±1σ 范围内？
- 是否存在异常极值点？

### 4. 时间一致性
- 训练集和测试集的表现是否一致？
- 因子特征是否在不同时期保持稳定？

---

## 清算压力因子（Liq_Zscore）示例

如果你的因子是清算压力因子，通过可视化可以验证：

```python
# Liq_Zscore 计算逻辑（参考）
# short_liq_ratio = liquidation_short / volume
# liq_log = log1p(short_liq_ratio)
# liq_zscore = (liq_log - rolling_mean) / rolling_std

# 可视化观察
diag.visualize_factor_vs_price(
    factor_name="Liq_Zscore",  # 你的清算压力因子名
    data_range='test',
    price_type='close',
    save_dir="./diagnostics"
)
```

**观察重点**：
- 价格见顶时，Liq_Zscore 是否有明显正向尖峰？
- 尖峰出现时间是否领先于价格下跌？
- 尖峰幅度是否超过 +1σ 边界？

---

## 保存的文件

图片自动保存为：`factor_vs_price_{因子名}_{train/test}.png`

例如：`factor_vs_price_gp_factor_0_test.png`

---

## 常见问题

**Q: 提示"因子不在 factor_data 中"？**  
A: 检查因子名称是否正确，可以先打印 `diag.selected_factors` 查看所有可用因子。

**Q: 图形不显示？**  
A: 设置 `save_dir` 参数将图片保存到文件，而不是显示。

**Q: 双 Y 轴刻度差异太大？**  
A: 这是正常的，因为价格和因子的量纲不同，重点关注趋势而非绝对数值。

**Q: 如何同时查看训练集和测试集？**  
A: 分别调用两次，设置 `data_range='train'` 和 `data_range='test'`。

