import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

print("=== p < 0.05 含义详解 ===\n")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. p值的基本概念
print("1. p值的基本概念:")
print("p值 = 在零假设为真的条件下，观察到当前结果或更极端结果的概率")
print()

print("2. 在相关性分析中的假设:")
print("零假设 (H0): 两个变量之间没有线性相关关系 (相关系数 = 0)")
print("备择假设 (H1): 两个变量之间存在线性相关关系 (相关系数 ≠ 0)")
print()

# 创建示例数据
np.random.seed(42)

# 情况1: 强相关关系
n = 50
x1 = np.random.normal(0, 1, n)
y1 = 0.8 * x1 + np.random.normal(0, 0.3, n)  # 强相关
corr1, p1 = pearsonr(x1, y1)

# 情况2: 弱相关关系
x2 = np.random.normal(0, 1, n)
y2 = 0.2 * x2 + np.random.normal(0, 1, n)  # 弱相关
corr2, p2 = pearsonr(x2, y2)

# 情况3: 无相关关系
x3 = np.random.normal(0, 1, n)
y3 = np.random.normal(0, 1, n)  # 无相关
corr3, p3 = pearsonr(x3, y3)

print("3. 不同相关性强度的p值示例:")
print(f"强相关: r={corr1:.4f}, p={p1:.6f} {'✓显著' if p1 < 0.05 else '✗不显著'}")
print(f"弱相关: r={corr2:.4f}, p={p2:.6f} {'✓显著' if p2 < 0.05 else '✗不显著'}")
print(f"无相关: r={corr3:.4f}, p={p3:.6f} {'✓显著' if p3 < 0.05 else '✗不显著'}")
print()

# 4. p < 0.05 的具体含义
print("4. p < 0.05 的具体含义:")
print("- 表示在零假设为真的情况下，出现当前观测结果的概率小于5%")
print("- 换句话说，这种结果很少是由纯粹的随机性造成的")
print("- 因此我们有95%的置信度认为相关性是真实存在的")
print("- 我们拒绝零假设，接受备择假设")
print()

print("5. 显著性水平的选择:")
print("α = 0.05 (5%): 最常用的标准")
print("α = 0.01 (1%): 更严格的标准，要求更强的证据")
print("α = 0.10 (10%): 较宽松的标准，在探索性研究中使用")
print()

# 6. 在你的置换检验代码中的应用
print("6. 在置换检验中的应用:")
print("你的代码:")
print("p_value = np.mean(np.abs(permuted_corrs) >= np.abs(original_corr))")
print()
print("这里的p值含义:")
print("- 表示在因子与收益率无关的假设下")
print("- 通过随机置换得到等于或大于观测相关系数的概率")
print("- 如果p < 0.05，说明观测到的相关性不太可能是随机产生的")
print("- 因此因子与收益率之间可能存在真实的关系")
print()

# 7. 实际示例计算
print("7. 实际示例 - 因子与收益率:")
factor = np.random.normal(0, 1, 100)
returns = 0.3 * factor + np.random.normal(0, 0.5, 100)

# 使用pearsonr计算
corr_direct, p_direct = pearsonr(factor, returns)

# 使用置换检验计算
n_permutations = 1000
original_corr = np.corrcoef(factor, returns)[0,1]
permuted_corrs = np.zeros(n_permutations)

for i in range(n_permutations):
    permuted_factor = np.random.permutation(factor)
    permuted_corrs[i] = np.corrcoef(permuted_factor, returns)[0,1]

p_permutation = np.mean(np.abs(permuted_corrs) >= np.abs(original_corr))

print(f"直接计算:")
print(f"  相关系数: {corr_direct:.4f}")
print(f"  p值: {p_direct:.6f}")
print(f"  是否显著: {'是' if p_direct < 0.05 else '否'}")
print()

print(f"置换检验:")
print(f"  相关系数: {original_corr:.4f}")
print(f"  p值: {p_permutation:.6f}")
print(f"  是否显著: {'是' if p_permutation < 0.05 else '否'}")
print()

# 8. p值的解释指南
print("8. p值解释指南:")
print("p < 0.001: 极强证据反对零假设 (***)")
print("p < 0.01:  强证据反对零假设 (**)")
print("p < 0.05:  中等证据反对零假设 (*)")
print("p < 0.10:  弱证据反对零假设")
print("p ≥ 0.10:  没有足够证据反对零假设")
print()

# 9. 注意事项
print("9. 使用p值的注意事项:")
print("- p值不能说明效应的大小，只能说明统计显著性")
print("- 小的p值不一定意味着实际意义重大")
print("- 大样本容易得到小p值，即使效应很小")
print("- p值不是效应真实存在的概率")
print("- 应该结合效应大小(如相关系数)一起解释")
print()

# 10. 在量化投资中的应用
print("10. 在量化投资中的意义:")
print("p < 0.05 表示:")
print("- 因子与收益率的关系不太可能是偶然的")
print("- 有95%的置信度认为这种关系是真实的")
print("- 可以将该因子纳入投资策略考虑")
print("- 但仍需要考虑实际经济意义和效应大小")

# 可视化p值的含义
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 左图：显著相关
ax1.scatter(x1, y1, alpha=0.6)
ax1.plot(np.sort(x1), 0.8 * np.sort(x1), 'r--', alpha=0.8)
ax1.set_title(f'显著相关\nr={corr1:.3f}, p={p1:.3f} < 0.05')
ax1.set_xlabel('因子值')
ax1.set_ylabel('收益率')

# 右图：不显著相关
ax2.scatter(x3, y3, alpha=0.6)
ax2.set_title(f'不显著相关\nr={corr3:.3f}, p={p3:.3f} > 0.05')
ax2.set_xlabel('因子值')
ax2.set_ylabel('收益率')

plt.tight_layout()
plt.show() 