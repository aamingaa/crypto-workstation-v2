做均值回归（Mean Reversion）策略，核心是捕捉**“不可持续的偏离”**。

在小时级别的Crypto合约市场，均值回归是非常有效的Alpha来源，但也是最容易“接飞刀”炸仓的策略。为了在gplearn中挖掘出鲁棒的因子，你需要将均值回归拆解为不同的维度，而不是只盯着价格看。

基于你之前的量化背景，以下是具体的实施路径，从**市场假设**到 **因子数学构造** 。

**一、 核心市场假设：为什么价格会回归？**

在设计因子前，必须先定义“回归”的物理驱动力，否则挖掘出的公式就是过拟合。

1. **流动性真空（Liquidity Holes）** ：

* **假设** ：短时间内大额市价单（Liquidation或巨鲸）吃光了Order Book的一侧深度，导致价格“虚高/虚低”。这种价格没有真实买卖盘支撑，会迅速回补。
* **现象** ：插针（Wicks）。

1. **过度拥挤的资金成本（Cost of Carry Constraints）** ：

* **假设** ：当资金费率（Funding Rate）极高时，持仓成本年化可能达到100%+。多头为了止损或规避费率会主动平仓，压低价格。
* **现象** ：费率达到极值后，价格往往向费率反方向移动。

1. **波动率均值回归（Volatility Mean Reversion）** ：

* **假设** ：极端的低波动（横盘）不可持续，必然爆发；极端的高波动（恐慌）不可持续，必然衰减。

**二、 因子拆分与数学构造 (Feature Engineering)**

在gplearn中，不要只扔进去一个 Close。你需要构建刻画上述假设的 **中间变量** （Primitives）。我们将均值回归因子拆分为三类： **价格类** 、 **费率/基差类** 、 **成交量类** 。

 *(* 以下公式使用类*gplearn*的算子逻辑， *ts_* 代表时序操作*)*

**1. 价格类回归 (Price Reversion)**

不要直接用RSI，要构建更纯粹的偏离度。

* Z-Score (标准分)：最经典的回归因子。

  $$
  Factor = \frac{Close - \text{ts\_mean}(Close, 24)}{\text{ts\_std}(Close, 24)}
  $$

  * 逻辑：价格偏离24小时均线2个标准差以上，大概率回调。
* 日内路径效率 (Path Efficiency Reversion)：

  $$
  Factor = \frac{Close - \text{ts\_delay}(Close, 24)}{\text{ts\_sum}(\text{abs}(Close - \text{ts\_delay}(Close, 1)), 24)}
  $$

  * 逻辑：如果过去24小时价格涨了10%，但路径非常曲折（分母大），说明多空博弈激烈，不容易反转。如果路径是一根直线拉上去（分母小），因子值极大，预示“超买”，大概率反转。

**2. 费率与情绪回归 (Sentiment Reversion) —— Crypto 特有**

这是Crypto最肥的Alpha来源。

* 费率偏离度 (Funding Divergence)：

  $$
  Factor = FundingRate - \text{ts\_mean}(FundingRate, 72)
  $$

  * 逻辑：当前的费率是否显著高于过去3天的平均水平？如果是，做空。
* 量价背离 (Price-OI Divergence)：

  $$
  Factor = \text{ts\_corr}(Close, OpenInterest, 24)
  $$

  * 逻辑：计算价格和持仓量的相关性。
  * 如果价格上涨 + OI下跌（相关性负），说明是空头止损带来的上涨（Short Squeeze），这种上涨缺乏新资金驱动，是做空的好机会（回归）。

**3. 市场微观结构回归 (Microstructure Reversion)**

* VWAP 乖离 (VWAP Deviation)：

  $$
  Factor = \frac{Close - VWAP_{24h}}{VWAP_{24h}}
  $$

  * 逻辑：VWAP是大资金的平均成本。价格大幅偏离VWAP意味着散户在追高，价格通常会回到大户成本线附近。
* 流动性冲击 (Liquidation Wick)：

  $$
  Factor = \frac{High - \text{max}(Open, Close)}{High - Low}
  $$

  * 逻辑：上影线长度占比。如果一根K线很长但实体很短（Factor值大），说明上方抛压极重（插针），下一根K线大概率下跌。

**三、 怎么用gplearn挖掘组合？**

有了上述基础算子，你可以让gplearn去挖掘它们的 **非线性组合** 。

**挖掘方向建议（Fitness Function导向）：**

1. **“条件”回归** ：单纯的RSI<30买入可能已经失效了。你希望gplearn挖掘出类似这样的逻辑：

* “当资金费率极高**且**价格快速拉升**且**成交量开始萎缩时，做空。”
* 公式示例：If(Funding > 0.05, -1 * ts_delta(Close, 1), 0)

1. **“环境”感知** ：

* 均值回归在**震荡市**赚钱，在**趋势市**亏钱。
* 你需要加入一个 **过滤器** （Filter）：比如 ADX 或者 ATR。
* 让gplearn学会：Factor = (MeanReversion_Signal) / (Trend_Strength + \epsilon)。趋势越强，回归信号权重越低。

**四、 避坑指南：如何让回归策略“鲁棒”？**

做均值回归最怕**“接飞刀”**（在单边暴跌中不断抄底）。

1. **必须有止损逻辑** ：在因子层面，如果偏离度过大（例如偏离均线5个标准差），不要认为是更大的机会，而要认为是 **市场结构崩塌** （黑天鹅），此时因子应该输出0或反向信号。
2. **左侧 vs 右侧** ：

* **左侧因子** （预测顶底）：利润高，胜率低。
* **右侧因子** （确认拐点）：利润低，胜率高。
* **建议** ：加入ts_delta算子。比如不要只看“价格高”，要看“价格高 + 价格开始拐头向下”那一刻。
* 公式：Condition = (Close > UpperBand) AND (ts_delta(Close, 1) < 0)

**总结**

对于小时级Crypto合约，最可靠的均值回归因子公式往往长这样：

$$
Alpha = -1 \times \underbrace{\text{ts\_rank}(\text{Close}, 24)}_{\text{价格高位}} \times \underbrace{(1 + \text{FundingRate})}_{\text{费率加成}} \times \underbrace{\frac{1}{\text{Volatility}}}_{\text{低波放大}}
$$

这个逻辑是：**在波动率低的时候，如果价格处于高位且费率很高，大力做空。**

你可以把这些作为种子（**Seeds**）喂给**gplearn**，让它去优化具体的参数和结构。
