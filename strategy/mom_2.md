### 动量（Momentum）相关描述（适合作为 `mom.md` 文档框架）

**动量策略的核心，是捕捉“趋势的持续性”，而不是预测拐点。** 在小时级别的 Crypto 永续合约市场，动量往往来自于：价格单边推进 + 真实资金持续进场 + 有利的波动环境。

---

### 一、核心市场假设：为什么价格会“继续朝一个方向走”？

1. **趋势惯性（Price Inertia）**

   - 假设：价格一旦在一段时间内单边上行/下行，说明市场形成了一致预期（Narrative），短期内不会立刻反转。
   - 现象：突破区间后的“顺势走一段”，而不是立刻打回区间。
2. **资金推动（Capital Inflow）**

   - 假设：只有当**新资金持续进场**（OI 增长、主动买盘占优），价格趋势才具有可持续性。
   - 现象：价格上涨 + OI 放大 + TakerBuy 明显大于 TakerSell，往往是“真趋势”。
3. **波动环境配合（Vol Regime）**

   - 假设：
     - 在长期低波动、短期突然放量突破时，动量延续概率大（趋势启动）。
     - 在本就极高波动的环境中突破，往往是假突破/短期情绪宣泄，容易反转。

---

### 二、因子拆分与数学构造（Feature Engineering）

为了让 gplearn 捕捉动量的不同维度，建议从 **价格强度、资金流/持仓、波动环境** 三个维度构造中间变量。

#### 1. 价格强度类（Price Strength Momentum）

- **超短期相对强度（Short-Term ROC Spread）**

\[
Factor_1 = \text{ts\_delta}(Close, \tau_1) - \text{ts\_delta}(Close, \tau_2)
\]

- 逻辑：比较短周期（如 \(\tau_1 = 4\) 小时）与中周期（如 \(\tau_2 = 12\) 小时）的涨跌幅差异；短期涨幅显著高于长期 → 动量在加速。
- **标准化价格动量（Vol-Normalized Momentum）**

\[
Factor_2 = \frac{Close - \text{ts\_delay}(Close, \tau)}{\text{ts\_std}(Return, \tau)}
\]

- 逻辑：收益率 ÷ 波动率，刻画“单位风险的趋势强度”；涨得多且稳（分母小） → 高质量趋势。
- **路径效率（Path Purity）**

\[
Factor_3 = \frac{\text{abs}(Close - \text{ts\_delay}(Close, 24))}{\text{ts\_sum}(\text{abs}(Close - \text{ts\_delay}(Close, 1)), 24)}
\]

- 逻辑：衡量过去 24 小时是“直线趋势”还是“来回震荡”；值接近 1 → 几乎直线拉升/下跌 → 纯净动量。

#### 2. 资金流与持仓类（Flow & OI Momentum）

- **增仓动量（Open Interest Momentum）**

\[
Factor_4 = \text{ts\_delta}(OpenInterest, 4) \times \text{sign}(\text{ts\_delta}(Close, 4))
\]

- 逻辑：

  - 价格上涨 + OI 上升 → 新资金顺势加仓，多头趋势真实且可持续；
  - 价格上涨 + OI 下降 → 空头回补主导，更偏“挤空反弹”，动量质量差。
- **主动流强度（Taker Flow Strength）**

\[
Factor_5 = \text{ts\_mean}(TakerBuyVolume - TakerSellVolume, 6)
\]

- 逻辑：过去 6 小时净主动买入（或卖出）的强度；持续净买入 → 趋势有微观资金支撑。

#### 3. 波动环境过滤（Vol-Regime Filter）

- **波动率过滤器（Vol Filter）**

\[
Filter = \frac{\text{ts\_std}(Return, 12)}{\text{ts\_std}(Return, 72)}
\]

- 逻辑：短期波动 / 长期波动：
  - \(Filter < 1\)：短期比长期更平静 → 突破时更可能走出**趋势行情**；
  - \(Filter \gg 1\)：短期波动已经极高 → 更容易是假突破/情绪尖峰。

---

### 三、怎么让 gplearn 挖掘“顺势 + 过滤假突破”的组合？

- **方向建议（Fitness 导向）**

  - 让 gplearn 学习类似逻辑：
    - “当价格动量强 **且** OI 增长 **且** 主动净买入为正 **且** 处于低波环境时，做多；
      当价格动量强 **且** OI 增长 **且** 主动净卖出为正时，做空。”
- **结构示例（给 gplearn 的 Seeds）**

\[
Alpha \approx \text{sign}(Factor_1) \times (1 + Factor_4) \times \frac{1}{1 + Filter}
\]

- 含义：
  - `sign(Factor_1)`：趋势方向；
  - `Factor_4`：资金是否跟随趋势；
  - `1 / (1 + Filter)`：在低波环境中放大动量信号，在高波环境中抑制。

---

### 四、避坑指南：如何让动量策略更鲁棒？

1. **防止追在“趋势尾巴”上**

   - 在因子层面加入 **“趋势年龄/累积涨幅”** 约束：
     - 累积涨幅过大、趋势持续时间过长 → 因子自动衰减，避免在尾声孤注一掷。
2. **识别假突破**

   - 价格突破但：
     - OI 不增反减，或
     - 主动买/卖净流不配合，
       则因子应降低权重甚至反向，避免被情绪尖刺扫掉。
3. **与均值回归因子的冲突处理**

   - 同一时间窗口内，动量因子与均值回归因子强烈对冲时：
     - 可以在组合层面引入 **Regime Switch / Meta-Labeling**：
       例如用 Vol Filter 或 Trend Filter 决定“此时优先信动量还是信回归”。

---

### 简短总结

- **价格强度**：回答“涨/跌得有多快、多稳？”
- **资金流与持仓**：回答“这波趋势背后有真金白银吗？”
- **波动环境**：回答“现在是容易走趋势，还是容易假突破？”

这样的三层拆分，非常适合交给 gplearn 做非线性组合，让它自动学习类似
**“if（低波 + 价格加速 + OI 增长 + 主动净买入）then（顺势加仓）”** 的高阶逻辑。
