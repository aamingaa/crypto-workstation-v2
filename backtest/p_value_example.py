import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子，确保结果可重复
np.random.seed(42)

# 创建示例数据
n_samples = 100
# 创建一个真实的因子（例如：市值因子）
factor = np.random.normal(0, 1, n_samples)
print(factor)

# 创建未来收益率，与因子有一定相关性
returns = 0.3 * factor + np.random.normal(0, 0.5, n_samples)

# [[1.0000, corr],
#  [corr,   1.0000]]

# 计算原始相关系数
original_corr = np.corrcoef(factor, returns)[0,1]

# print(original_corr)

# 进行100次置换
n_permutations = 100
permuted_corrs = np.zeros(n_permutations)

# 进行置换并计算相关系数
for i in range(n_permutations):
    permuted_factor = np.random.permutation(factor)
    permuted_corrs[i] = np.corrcoef(permuted_factor, returns)[0,1]

# 计算p值
p_value = np.mean(np.abs(permuted_corrs) >= np.abs(original_corr))

print(f"原始相关系数: {original_corr:.4f}")
print(f"置换检验p值: {p_value:.4f}")
print(f"是否显著（p < 0.05）: {p_value < 0.05}")

# 设置中文字体，避免显示乱码
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']  # 按优先级尝试不同字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 可视化
plt.figure(figsize=(10, 6))
plt.hist(np.abs(permuted_corrs), bins=30, alpha=0.5, label='随机置换相关系数绝对值分布')
plt.axvline(np.abs(original_corr), color='r', linestyle='--', 
           label=f'原始相关系数绝对值 ({np.abs(original_corr):.4f})')
plt.title(f'置换检验结果 (p值 = {p_value:.4f})')
plt.xlabel('相关系数绝对值')
plt.ylabel('频数')
plt.legend()
plt.grid(True, alpha=0.3)

# 打印详细的统计信息
print("\n详细统计信息：")
print(f"置换后相关系数绝对值的均值: {np.mean(np.abs(permuted_corrs)):.4f}")
print(f"置换后相关系数绝对值的标准差: {np.std(np.abs(permuted_corrs)):.4f}")
print(f"置换后相关系数绝对值的最大值: {np.max(np.abs(permuted_corrs)):.4f}")
print(f"置换后相关系数绝对值的最小值: {np.min(np.abs(permuted_corrs)):.4f}")
print(f"大于等于原始相关系数绝对值的次数: {np.sum(np.abs(permuted_corrs) >= np.abs(original_corr))}")
print(f"总置换次数: {n_permutations}")

plt.show() 