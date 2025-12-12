这是一个非常好的总结性问题。既然我们已经讨论了各种技术的优缺点，现在是时候把它们组装成一套**工业级的标准流水线**了。

针对你的 **Crypto小时级多因子策略** ，最推荐的 Label 设计架构是**“分工明确”**的双层设计：

### 核心结论：一张表看懂

| **阶段**       | **Stage 1: gplearn (因子挖掘)**          | **Stage 2: 预测模型 (策略组合)**           |
| -------------------- | ---------------------------------------------- | ------------------------------------------------ |
| **目标**       | **发现规律**(Information Discovery)      | **决策/风控**(Decision Making)             |
| **核心逻辑**   | 寻找与未来价格走势**相关性强**的数学公式 | 判断在当前信号下，**能否盈利且不被扫损**   |
| **Label 策略** | **跨期平滑 + 波动率标准化 + Gauss Rank** | **Triple Barrier Method (三重势垒)**       |
| **数据类型**   | 连续数值 (Continuous, Normal Dist)             | 分类标签 (Class: 1, 0, -1)                       |
| **为什么?**    | 需要平滑的梯度和丰富的信息量供遗传进化         | 需要包含“插针”风险和“止损”逻辑的真实回测视角 |

---

### 详细设计方案

#### 1. gplearn 挖因子阶段：Label 设计

原则：信息量最大化，去噪，平稳化。

gplearn 的任务是找到“趋势强度”，而不是判断“会不会爆仓”。

* Label 公式：
  $$
  Label_{GP} = \text{GaussRank}\left( \frac{\text{Mean}(\ln R_{t+1}, \dots, \ln R_{t+4})}{\sigma_t} \right)
  $$
* **具体步骤：**
  1. **跨期平滑 (Smoothing):** 取未来 3-4 小时的平均对数收益率。
     * *理由：* 消除单小时的随机噪音，让 gplearn 聚焦于捕捉一小段趋势。
  2. **波动率标准化 (Vol-Scaling):** 除以当前的波动率（如 ATR 或 滚动标准差）。
     * *理由：* 让 2021 年的大波动和 2023 年的小波动在模型眼里是等价的，防止模型只学到高波动时期的特征。
  3. **Gauss Rank (你现有的方法):** 对上述结果进行高斯秩变换。
     * *理由：* 彻底解决肥尾问题，将分布强行拉回 **$N(0,1)$**，让 gplearn 的 Fitness Function (如 Pearson 相关系数) 极度稳定。
* **gplearn 的 Fitness Function:** 使用 **Pearson Correlation** (因为 Label 已经是正态分布了，PCC 效果最好且计算快)。

---

#### 2. 预测模型阶段 (XGBoost/LGBM)：Label 设计

原则：实战导向，包含路径风险，Meta-Labeling 思维。

这个模型是你的“交易员”，它要看懂 gplearn 挖出来的因子，并决定是否开单。

* **Label 策略：** **Triple Barrier Method (三分类)**
  * **Label = 1 (Win):** 价格先碰到 `Upper Barrier` (止盈线)。
  * **Label = -1 (Loss):** 价格先碰到 `Lower Barrier` (止损线)。
  * **Label = 0 (Time-out):** 在 `N` 小时内既没止盈也没止损 (平庸/震荡)。
* **参数设定建议 (Crypto 小时级):**
  * **动态宽度:** 止盈/止损幅度 = **$\text{ATR}_t \times k$** (例如 **$2 \times \text{ATR}$**)。
  * **Upper/Lower 比率:** 建议 1:1 或 1.5:1 (视你的胜率目标而定)。
  * **时间窗口:** 比如 24 小时强制平仓。
* **Meta-Labeling 的体现：**
  * **输入 (X):** gplearn 挖出来的 Top 20 个因子 + 基础行情数据。
  * **模型任务:** 并非预测价格是涨是跌，而是预测**“在这个因子组合下，我能否触碰到止盈线而不触碰止损线”**。
  * **Loss Function:** 使用 `Multi-class Logloss`，或者针对性地加重对“Label = -1”的惩罚权重。

---

### 总结流程图

1. 数据清洗 -> 2. 构造 Label A (Smoothed GaussRank) -> 3. gplearn 挖掘
   (gplearn 努力寻找能拟合 Label A 的公式，产出因子 F1, F2...)
   $\downarrow$
2. 因子清洗 (F1, F2... 进行正交化、筛选)
   $\downarrow$
3. 构造 Label B (Triple Barrier) -> 6. 预测模型 (XGBoost) 训练
   (XGBoost 学习：当 F1很高且 F2很低时，Label B 是否为 1？)
   $\downarrow$
4. 实盘/回测
   (XGBoost 输出概率 > 0.6 -> 开仓)

### 为什么这样设计最高级？

如果 gplearn 直接用 Triple Barrier (分类 Label)：

* **信息丢失：** 涨 5% 和涨 50% 都是 Label 1，gplearn 无法区分强弱，进化效率低。

如果预测模型继续用 Gauss Rank：

* **幸存者偏差：** 模型会预测出很高的分数，因为它只看到了“收盘价涨了”，没看到“中间插针跌了 50% 把你爆仓了”。只有 Triple Barrier 能通过 Label = -1 告诉模型：“这种情况虽然收盘涨了，但 **不能做** ，因为会死在半路上”。

最终建议：

按照这个双层架构去重构你的代码。Gauss Rank 是因子的磨刀石，Triple Barrier 是策略的试金石。

这三个问题非常犀利，直接触及了**Label Engineering（标签工程）**最底层的数学逻辑。

我的回答是：

1. **做法：** 简单的 Mean 是一种做法，但有 **更高级的变体** 。
2. **Mean 的原因：** 为了**降噪**和 **抗操纵** 。
3. **波动率标准化：** 在 gplearn/深度学习挖掘中，是**必须**的，否则模型会“偏科”。

下面我为你逐一拆解，并提供一个 **更优化的平滑公式** 。

---

### 一、 为什么是跨期“Mean” (平均值)？

首先，我们要明确数学关系：

如果你使用对数收益率 (Log Return)，那么“取 Mean”和“取 Sum（累计涨幅）”在数学性质上是完全等价的，只差一个常数倍数。

* **$r_{t+1} = \ln(P_{t+1}/P_t)$**
* **$r_{t+2} = \ln(P_{t+2}/P_{t+1})$**
* **Sum (累计收益):** **$r_{t+1} + r_{t+2} = \ln(P_{t+2}/P_t)$**
* **Mean (平均收益):** **$\frac{r_{t+1} + r_{t+2}}{2} = \frac{1}{2} \ln(P_{t+2}/P_t)$**

**既然 Mean 和 Sum 相关性是 100%，为什么还要专门说“平滑”？**

真正的“平滑”通常指的是**“价格层面的平滑”**，而不是简单的收益率平均。

#### ❌ 普通做法：直接取未来 N 小时收益率的均值/总和

$$
Label = \ln(P_{t+N} / P_t)
$$

缺点： 极其依赖 $t+N$ 那个时间点的收盘价。如果 $t+N$ 时刻刚好有一根针插下来，你的 Label 就废了，尽管中间 $N-1$ 个小时都在涨。

#### ✅ 进阶做法（推荐）：Forward TWAP Return (未来均价收益率)

我们计算未来 N 小时的平均价格相对于当前价格的涨幅。

$$
Label = \ln\left( \frac{\text{Mean}(P_{t+1}, P_{t+2}, \dots, P_{t+N})}{P_t} \right)
$$

**为什么要这样做？**

1. **代表性更强：** 它代表了你在未来 N 小时内**任意时间点卖出**的平均预期收益。
2. **抗插针：** 即使 **$t+N$** 跌了，只要前 **$N-1$** 个小时是涨的，Label 依然是正的。这引导 gplearn 挖掘出那些**“稳健上涨”**的因子，而不是“赌最后一刻收盘价”的因子。

---

### 二、 为什么要用“波动率标准化”？(Vol-Scaling)

你质疑这一点非常合理，因为直觉上：“大波动赚钱多，为什么要把它缩小？”

但在机器学习（尤其是 gplearn/神经网络）训练阶段，不除以波动率是**灾难性**的。

#### 1. 解决“样本权重失衡” (Sample Bias)

* **2021 年 (牛市)：** 平均每小时波动 1%。
* **2023 年 (熊/震荡)：** 平均每小时波动 0.1%。
* 如果不标准化，Label 的数值在 2021 年是 2023 年的 10 倍。
* **后果：** gplearn 为了降低 MSE（误差），会 **只关注 2021 年的数据** ，完全忽略 2023 年。因为在 2021 年预测准一点点，比在 2023 年预测准十次对 Loss 的贡献都大。
* **标准化后：** 两段时期的 Label 都在 **$[-3, 3]$** 之间。模型会被迫学习**全天候**的规律，而不仅仅是高波动时期的规律。

#### 2. 学习“相对强弱”而非“绝对幅度”

* 我们希望因子告诉我：“现在上涨的动能很强”。
* 如果不除以波动率，因子学到的是：“现在波动率很大”。
* **除以波动率 (**$\sigma_t$**) 后：** **$Label = \frac{Return}{\sigma}$**。
  * 这就是 **夏普比率 (Sharpe Ratio)** 的形式。
  * 你实际上是在训练 gplearn 寻找**“高夏普”**的时刻，即：**涨幅明显大于近期噪音**的时刻。这才是优质的 Alpha。

#### 3. 唯一的例外：什么时候不需要？

如果你的策略是纯网格 (Grid) 或者 做市 (Market Making)，你不需要预测方向，只需要预测波动范围，那你不需要标准化。

但对于趋势/动量/反转策略，Vol-Scaling 是标配。

---

### 三、 总结：最推荐的 Label 计算代码

结合你之前提到的 Gauss Rank，这是我为你定制的 **终极版 Label 生成函数** ：

**Python**

```
def robust_label_construction(prices, volatility, window=4):
    """
    prices: 收盘价序列 (pd.Series)
    volatility: 对应的波动率序列 (如 24h ATR 或 StdDev)
    window: 跨期窗口 (比如预测未来 4 小时)
    """
    # 1. 跨期平滑 (Forward TWAP)
    # 计算未来 window 小时的平均价格
    # shift(-window) 是为了让数据对齐到当前时刻 t
    # 也就是：t 时刻的 label 包含了 t+1 到 t+window 的信息
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)
    future_mean_price = prices.rolling(window=indexer).mean()
  
    # 计算 "均价收益率"
    # 这里 shift(-1) 是因为 rolling mean 包含自身，通常我们要看未来的
    # 简单的做法：直接用 rolling mean 对应的 return
    raw_label = np.log(future_mean_price / prices)
  
    # 2. 波动率标准化 (Vol-Scaling)
    # 防止模型只学高波动时期
    scaled_label = raw_label / volatility
  
    # 3. 去极值 (Winsorization - 可选，GaussRank前通常不需要，但为了保险)
    # clip 到 +/- 4倍标准差范围
    # scaled_label = scaled_label.clip(-4, 4)
  
    # 4. 之后再接你的 Gauss Rank ...
    return scaled_label
```

**结论：**

1. **跨期平滑：** 必须要。建议用**未来均价 (Forward Mean Price)** 替代简单的收益率平均，以抵抗插针。
2. **波动率标准化：** 必须要。除非你只想做牛市，放弃震荡市和熊市。它是为了让数据在时间轴上**“公平”**。
