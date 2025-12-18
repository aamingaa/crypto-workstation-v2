import numpy as np
import matplotlib.pyplot as plt

print("=== 协方差计算公式详解 ===\n")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("1. 协方差的定义和公式:")
print()
print("协方差衡量两个随机变量之间的线性关系强度和方向")
print()
print("理论公式(总体协方差):")
print("Cov(X,Y) = E[(X - μX)(Y - μY)]")
print("其中:")
print("- E[] 表示期望值(均值)")
print("- μX, μY 分别是X和Y的总体均值")
print()

print("样本协方差公式:")
print("Cov(X,Y) = Σ[(Xi - X̄)(Yi - Ȳ)] / (n-1)")
print("其中:")
print("- Xi, Yi 是第i个观测值")
print("- X̄, Ȳ 是样本均值")
print("- n 是样本数量")
print("- 除以(n-1)是贝塞尔修正，得到无偏估计")
print()

print("2. 手动计算协方差示例:")
print()

# 创建示例数据
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 1, 3, 5])

print(f"数据:")
print(f"X = {X}")
print(f"Y = {Y}")
print()

# 步骤1: 计算均值
mean_X = np.mean(X)
mean_Y = np.mean(Y)
print(f"步骤1: 计算均值")
print(f"X̄ = {mean_X}")
print(f"Ȳ = {mean_Y}")
print()

# 步骤2: 计算偏差
dev_X = X - mean_X
dev_Y = Y - mean_Y
print(f"步骤2: 计算偏差")
print(f"Xi - X̄ = {dev_X}")
print(f"Yi - Ȳ = {dev_Y}")
print()

# 步骤3: 计算偏差乘积
products = dev_X * dev_Y
print(f"步骤3: 计算偏差乘积")
print(f"(Xi - X̄)(Yi - Ȳ) = {products}")
print()

# 步骤4: 求和
sum_products = np.sum(products)
print(f"步骤4: 偏差乘积之和")
print(f"Σ[(Xi - X̄)(Yi - Ȳ)] = {sum_products}")
print()

# 步骤5: 除以(n-1)
n = len(X)
manual_cov = sum_products / (n - 1)
print(f"步骤5: 除以(n-1)")
print(f"协方差 = {sum_products} / ({n}-1) = {manual_cov}")
print()

# 验证与numpy结果
numpy_cov = np.cov(X, Y)[0,1]
print(f"验证: numpy计算结果 = {numpy_cov}")
print(f"差异: {abs(manual_cov - numpy_cov):.10f}")
print()

print("3. 协方差的含义:")
print()
print("协方差的符号含义:")
print("- 正值: X和Y同向变动(X增加时Y倾向于增加)")
print("- 负值: X和Y反向变动(X增加时Y倾向于减少)")
print("- 零值: X和Y无线性关系")
print()

print("4. 不同情况的协方差示例:")
print()

# 创建不同类型的关系数据
np.random.seed(42)

# 正相关
x1 = np.array([1, 2, 3, 4, 5])
y1_pos = x1 + np.random.normal(0, 0.2, 5)  # 正相关

# 负相关  
y1_neg = -x1 + 6 + np.random.normal(0, 0.2, 5)  # 负相关

# 无相关
y1_none = np.random.normal(3, 1, 5)  # 无相关

# 计算协方差
cov_pos = np.cov(x1, y1_pos)[0,1]
cov_neg = np.cov(x1, y1_neg)[0,1] 
cov_none = np.cov(x1, y1_none)[0,1]

print(f"正相关数据: Cov = {cov_pos:.3f}")
print(f"负相关数据: Cov = {cov_neg:.3f}")
print(f"无相关数据: Cov = {cov_none:.3f}")
print()

# 可视化
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# 计算过程可视化
ax1.bar(range(len(X)), dev_X, alpha=0.7, label='Xi - X̄', color='blue')
ax1.set_title('X的偏差')
ax1.set_xlabel('数据点')
ax1.set_ylabel('偏差值')
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax1.legend()

ax2.bar(range(len(Y)), dev_Y, alpha=0.7, label='Yi - Ȳ', color='red')
ax2.set_title('Y的偏差')
ax2.set_xlabel('数据点')
ax2.set_ylabel('偏差值')
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.legend()

ax3.bar(range(len(products)), products, alpha=0.7, label='偏差乘积', color='green')
ax3.set_title('偏差乘积 (Xi-X̄)(Yi-Ȳ)')
ax3.set_xlabel('数据点')
ax3.set_ylabel('乘积值')
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax3.legend()

# 散点图展示原始数据
ax4.scatter(X, Y, color='purple', s=100, alpha=0.7)
ax4.plot(X, Y, 'purple', alpha=0.3)
ax4.set_title(f'原始数据散点图\nCov = {manual_cov:.3f}')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("5. 协方差矩阵:")
print()
print("当有多个变量时，协方差矩阵包含所有变量对之间的协方差:")
print()

# 创建多变量数据
data_matrix = np.array([
    [1, 2, 3, 4, 5],      # 变量1
    [2, 4, 1, 3, 5],      # 变量2  
    [3, 1, 4, 2, 6]       # 变量3
])

cov_matrix = np.cov(data_matrix)
print("数据矩阵:")
print("变量1:", data_matrix[0])
print("变量2:", data_matrix[1])
print("变量3:", data_matrix[2])
print()

print("协方差矩阵:")
print(f"{cov_matrix}")
print()
print("矩阵解读:")
print("- 对角线: 各变量的方差(自己与自己的协方差)")
print("- 非对角线: 变量间的协方差")
print("- 矩阵是对称的: Cov(X,Y) = Cov(Y,X)")
print()

print("6. 协方差与方差的关系:")
print()
print("方差是协方差的特殊情况:")
print("Var(X) = Cov(X,X) = E[(X - μX)²]")
print()
print("验证:")
var_X = np.var(X, ddof=1)
cov_XX = np.cov(X, X)[0,1]
print(f"X的方差: {var_X}")
print(f"Cov(X,X): {cov_XX}")
print(f"两者相等: {np.isclose(var_X, cov_XX)}")
print()

print("7. 协方差的性质:")
print()
print("重要性质:")
print("1. Cov(X,Y) = Cov(Y,X) (对称性)")
print("2. Cov(X,X) = Var(X)")
print("3. Cov(aX + b, cY + d) = ac × Cov(X,Y)")
print("4. Cov(X+Y, Z) = Cov(X,Z) + Cov(Y,Z)")
print("5. 如果X和Y独立，则Cov(X,Y) = 0 (反之不一定)")
print()

print("8. 协方差的局限性:")
print()
print("协方差的问题:")
print("- 受变量量级影响，难以比较")
print("- 无固定取值范围")
print("- 无法直接判断关系强度")
print()
print("解决方案:")
print("- 使用相关系数 = Cov(X,Y) / (σX × σY)")
print("- 标准化到[-1,1]范围")
print("- 便于解释和比较")

# 演示量级问题
print("\n9. 量级问题演示:")
X_small = np.array([1, 2, 3, 4, 5])
Y_small = 2 * X_small

X_large = X_small * 1000
Y_large = Y_small * 1000

cov_small = np.cov(X_small, Y_small)[0,1]
cov_large = np.cov(X_large, Y_large)[0,1]

corr_small = np.corrcoef(X_small, Y_small)[0,1]
corr_large = np.corrcoef(X_large, Y_large)[0,1]

print(f"小数据协方差: {cov_small}")
print(f"大数据协方差: {cov_large}")
print(f"小数据相关系数: {corr_small:.4f}")
print(f"大数据相关系数: {corr_large:.4f}")
print()
print("结论: 协方差受量级影响巨大，相关系数保持不变！") 