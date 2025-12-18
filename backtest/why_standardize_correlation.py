import numpy as np
import matplotlib.pyplot as plt

print("=== 为什么相关系数要用协方差除以标准差乘积 ===\n")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("1. 相关系数公式回顾:")
print("相关系数 r = Cov(X,Y) / (σX × σY)")
print("其中:")
print("- Cov(X,Y) 是协方差，衡量两变量共同变动")
print("- σX, σY 是标准差，衡量各自的波动程度")
print()

# 创建示例数据来说明问题
np.random.seed(42)

print("2. 为什么需要标准化？让我们看几个例子:\n")

# 例子1: 相同关系，不同量级
print("例子1: 相同线性关系，不同数据量级")
x1 = np.array([1, 2, 3, 4, 5])
y1 = 2 * x1  # 完全线性关系

x2 = np.array([100, 200, 300, 400, 500])  # 放大100倍
y2 = 2 * x2  # 同样的线性关系

# 计算协方差
cov1 = np.cov(x1, y1)[0,1]
cov2 = np.cov(x2, y2)[0,1]

# 计算标准差
std_x1, std_y1 = np.std(x1, ddof=1), np.std(y1, ddof=1)
std_x2, std_y2 = np.std(x2, ddof=1), np.std(y2, ddof=1)

# 计算相关系数
corr1 = cov1 / (std_x1 * std_y1)
corr2 = cov2 / (std_x2 * std_y2)

print(f"小数据: x={x1}, y={y1}")
print(f"协方差: {cov1:.2f}")
print(f"标准差: σx={std_x1:.2f}, σy={std_y1:.2f}")
print(f"相关系数: {corr1:.4f}")
print()

print(f"大数据: x={x2}, y={y2}")
print(f"协方差: {cov2:.2f}")
print(f"标准差: σx={std_x2:.2f}, σy={std_y2:.2f}")
print(f"相关系数: {corr2:.4f}")
print()

print("观察结果:")
print(f"- 协方差差异巨大: {cov1:.0f} vs {cov2:.0f}")
print(f"- 但相关系数相同: {corr1:.4f} vs {corr2:.4f}")
print("这说明相关系数消除了量级的影响!\n")

# 例子2: 协方差的问题
print("例子2: 为什么协方差不够用？")

# 创建三组数据，相同的线性关系但不同波动性
x = np.array([1, 2, 3, 4, 5])
y_low = x + 0.1 * np.random.randn(5)     # 低噪声
y_med = x + 0.5 * np.random.randn(5)     # 中等噪声  
y_high = x + 2.0 * np.random.randn(5)    # 高噪声

cov_low = np.cov(x, y_low)[0,1]
cov_med = np.cov(x, y_med)[0,1]
cov_high = np.cov(x, y_high)[0,1]

corr_low = np.corrcoef(x, y_low)[0,1]
corr_med = np.corrcoef(x, y_med)[0,1]
corr_high = np.corrcoef(x, y_high)[0,1]

print(f"低噪声: 协方差={cov_low:.3f}, 相关系数={corr_low:.3f}")
print(f"中噪声: 协方差={cov_med:.3f}, 相关系数={corr_med:.3f}")
print(f"高噪声: 协方差={cov_high:.3f}, 相关系数={corr_high:.3f}")
print()

print("协方差的问题:")
print("- 协方差受数据波动性影响")
print("- 无法直接比较不同数据集的关系强度")
print("- 相关系数通过标准化解决了这个问题\n")

# 3. 标准化的数学原理
print("3. 标准化的数学原理:")
print()
print("协方差公式: Cov(X,Y) = E[(X-μX)(Y-μY)]")
print("问题: 协方差的值依赖于X和Y的量级")
print()
print("解决方案: 将X和Y都标准化为标准分数")
print("标准分数: Z = (X - μX) / σX")
print()
print("标准化后:")
print("r = Cov(ZX, ZY) = E[ZX × ZY]")
print("  = E[((X-μX)/σX) × ((Y-μY)/σY)]")
print("  = E[(X-μX)(Y-μY)] / (σX × σY)")
print("  = Cov(X,Y) / (σX × σY)")
print()

# 4. 可视化演示
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# 原始数据对比
ax1.scatter(x1, y1, color='blue', s=50)
ax1.plot(x1, y1, 'b--', alpha=0.7)
ax1.set_title(f'小量级数据\nCov={cov1:.1f}, r={corr1:.3f}')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

ax2.scatter(x2, y2, color='red', s=50)
ax2.plot(x2, y2, 'r--', alpha=0.7)
ax2.set_title(f'大量级数据\nCov={cov2:.0f}, r={corr2:.3f}')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

# 标准化后的数据
x1_std = (x1 - np.mean(x1)) / np.std(x1, ddof=1)
y1_std = (y1 - np.mean(y1)) / np.std(y1, ddof=1)
x2_std = (x2 - np.mean(x2)) / np.std(x2, ddof=1)
y2_std = (y2 - np.mean(y2)) / np.std(y2, ddof=1)

ax3.scatter(x1_std, y1_std, color='blue', s=50)
ax3.plot(x1_std, y1_std, 'b--', alpha=0.7)
ax3.set_title('标准化后的小量级数据')
ax3.set_xlabel('标准化 X')
ax3.set_ylabel('标准化 Y')

ax4.scatter(x2_std, y2_std, color='red', s=50)
ax4.plot(x2_std, y2_std, 'r--', alpha=0.7)
ax4.set_title('标准化后的大量级数据')
ax4.set_xlabel('标准化 X')
ax4.set_ylabel('标准化 Y')

plt.tight_layout()
plt.show()

# 5. 在量化投资中的重要性
print("5. 在量化投资中的重要性:")
print()
print("为什么标准化对IC计算很重要:")
print()

# 模拟真实投资场景
print("场景: 比较不同因子的预测能力")
# 因子1: PE比率 (范围: 5-50)
pe_ratio = np.array([8, 12, 25, 35, 15])
returns1 = np.array([0.15, 0.08, -0.05, -0.12, 0.06])

# 因子2: 市值 (范围: 100亿-5000亿)
market_cap = np.array([150, 800, 2500, 4200, 600])  # 亿元
returns2 = np.array([0.12, 0.05, -0.03, -0.08, 0.08])

# 计算协方差和相关系数
cov_pe = np.cov(pe_ratio, returns1)[0,1]
cov_cap = np.cov(market_cap, returns2)[0,1]

ic_pe = np.corrcoef(pe_ratio, returns1)[0,1]
ic_cap = np.corrcoef(market_cap, returns2)[0,1]

print(f"PE因子:")
print(f"  数据范围: {pe_ratio.min()}-{pe_ratio.max()}")
print(f"  协方差: {cov_pe:.6f}")
print(f"  IC值: {ic_pe:.4f}")
print()

print(f"市值因子:")
print(f"  数据范围: {market_cap.min()}-{market_cap.max()}")  
print(f"  协方差: {cov_cap:.6f}")
print(f"  IC值: {ic_cap:.4f}")
print()

print("结论:")
print("- 如果只看协方差，无法比较两个因子的预测能力")
print("- IC值(相关系数)消除了量级差异，可以直接比较")
print("- 这就是为什么在因子分析中使用IC而不是协方差")

print("\n6. 总结:")
print("除以标准差乘积的作用:")
print("1. 标准化: 消除量级影响，使结果在[-1,1]范围内")
print("2. 可比性: 不同数据集的相关系数可以直接比较")
print("3. 解释性: 值的大小直接反映关系强度")
print("4. 稳健性: 不受数据单位或缩放影响")
print()
print("这就是为什么相关系数 = 协方差 / (标准差1 × 标准差2)！") 