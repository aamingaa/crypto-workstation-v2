"""
测试 tsfresh 提取持仓量特征的完整示例
"""
import pandas as pd
from gp_crypto_next.oi_dataload import build_oi_features_tsfresh, build_oi_features_tsfresh_custom

# ============ 步骤1: 加载数据 ============
# 假设你的数据从 CSV 或其他方式加载
# df_raw = pd.read_csv('your_data.csv')

# 示例：如果你的数据格式和截图一样
def prepare_oi_data(df_raw):
    """
    准备持仓量数据用于 tsfresh 特征提取
    
    输入: 原始数据，包含 open_time 和 openInterest 列
    输出: 设置好索引的 DataFrame
    """
    df = df_raw.copy()
    
    # 确保 open_time 是 datetime 类型
    if not pd.api.types.is_datetime64_any_dtype(df['open_time']):
        df['open_time'] = pd.to_datetime(df['open_time'])
    
    # 设置时间索引
    df.set_index('open_time', inplace=True)
    
    # 排序（确保时间顺序）
    df.sort_index(inplace=True)
    
    print(f"数据范围: {df.index.min()} 到 {df.index.max()}")
    print(f"数据点数: {len(df)}")
    print(f"列: {df.columns.tolist()}")
    
    return df


# ============ 步骤2: 提取特征 ============
def extract_15min_features(df_sec, method='custom'):
    """
    从秒级数据提取15分钟级别特征
    
    参数:
        df_sec: 秒级数据，index为时间戳，包含'openInterest'列
        method: 'minimal', 'custom', 'comprehensive'
    """
    print(f"\n开始使用 {method} 方法提取特征...")
    
    if method == 'minimal':
        features = build_oi_features_tsfresh(df_sec, feature_params='minimal')
    elif method == 'custom':
        features = build_oi_features_tsfresh_custom(df_sec)
    elif method == 'comprehensive':
        features = build_oi_features_tsfresh(df_sec, feature_params='comprehensive')
    else:
        raise ValueError(f"未知方法: {method}")
    
    print(f"✓ 提取完成！")
    print(f"  - 生成 {len(features)} 个15分钟时间窗口")
    print(f"  - 每个窗口有 {len(features.columns)} 个特征")
    print(f"  - 特征维度: {features.shape}")
    print(f"  - 缺失值: {features.isna().sum().sum()}")
    
    return features


# ============ 步骤3: 检查结果 ============
def analyze_features(features):
    """分析提取的特征"""
    print("\n" + "="*60)
    print("特征分析")
    print("="*60)
    
    # 显示前几个特征
    print("\n前5个特征列:")
    print(features.columns[:5].tolist())
    
    # 显示统计信息
    print("\n特征统计:")
    print(features.describe().iloc[:, :3])  # 只显示前3列
    
    # 检查哪些特征有缺失值
    missing = features.isna().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"\n有缺失值的特征 ({len(missing)} 个):")
        print(missing.head(10))
    else:
        print("\n✓ 所有特征都没有缺失值！")
    
    # 显示时间范围
    print(f"\n时间窗口范围:")
    print(f"  开始: {features.index.min()}")
    print(f"  结束: {features.index.max()}")
    print(f"  间隔: 15分钟")


# ============ 主函数 ============
if __name__ == "__main__":
    
    # 示例1: 如果你有 CSV 文件
    # df_raw = pd.read_csv('path/to/your/oi_data.csv')
    # df_sec = prepare_oi_data(df_raw)
    
    # 示例2: 如果你的数据已经在内存中（比如从 notebook）
    # 假设你已经有了 df_sec，且格式如下：
    # - index: DatetimeIndex (open_time)
    # - columns: ['openInterest']
    
    print("请确保你的 df_sec 格式如下:")
    print("  - index: DatetimeIndex (时间戳)")
    print("  - columns: ['openInterest']")
    print("\n如果数据格式正确，执行以下代码:")
    print("-" * 60)
    print("""
# 方法1: 使用定制化特征（推荐，适合金融数据）
features_custom = build_oi_features_tsfresh_custom(df_sec)

# 方法2: 使用极简模式（快速测试）
features_minimal = build_oi_features_tsfresh(df_sec, feature_params='minimal')

# 方法3: 使用全面模式（特征最多，但计算慢）
# features_comprehensive = build_oi_features_tsfresh(df_sec, feature_params='comprehensive')

# 查看结果
print(features_custom.shape)
print(features_custom.head())

# 保存结果
features_custom.to_csv('oi_features_15min.csv')
    """)
    
    print("-" * 60)
    print("\n完整使用示例:")
    print("""
# 在你的 notebook 或脚本中：
from gp_crypto_next.oi_dataload import build_oi_features_tsfresh_custom
import pandas as pd

# 1. 加载你的秒级数据
df_raw = pd.read_csv('your_oi_data.csv')
df_raw['open_time'] = pd.to_datetime(df_raw['open_time'])
df_sec = df_raw.set_index('open_time').sort_index()

# 2. 提取15分钟特征
features = build_oi_features_tsfresh_custom(df_sec)

# 3. 查看结果
print(f"提取了 {features.shape[1]} 个特征")
print(f"时间窗口数: {len(features)}")
print(features.head())

# 4. 后续处理
# - 可以和你的15分钟K线数据合并
# - 用于机器学习模型训练
    """)

