import numpy as np
import pandas as pd
import logging
from tsfresh import extract_features
from scipy.stats import linregress

# 配置日志（方便排查问题）
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 兼容不同tsfresh版本的impute函数
try:
    from tsfresh.utilities.dataframe_functions import impute
except ImportError:
    def impute(df: pd.DataFrame) -> pd.DataFrame:
        """自定义简易填充函数，替代tsfresh的impute"""
        for col in df.columns:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                df[col] = df[col].fillna(df[col].median())
        return df

def calculate_rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """计算滚动窗口内的线性回归斜率（反映短期趋势）"""
    def _slope(arr):
        if len(arr) < 2:
            return np.nan
        x = np.arange(len(arr))
        return linregress(x, arr).slope
    
    return series.rolling(window=window, min_periods=2).apply(_slope, raw=True)

def build_oi_features_tsfresh_custom(
    df_sec: pd.DataFrame,
    window_freq: str = '15min',
    n_jobs: int = 4,
    disable_progressbar: bool = True,
    chunksize: int = None,
    rolling_window_sizes: list = [5, 10, 20]  # 滚动窗口大小（按数据周期数，如1分钟数据则5=5分钟）
) -> pd.DataFrame:
    """
    针对持仓量数据的定制化 tsfresh 特征提取（新增滚动窗口特征）
    结合原有逻辑 + tsfresh优势 + 滚动窗口动态特征
    
    参数:
        df_sec: 输入DataFrame，需包含'openInterest'列，索引为datetime类型（推荐naive datetime）
        window_freq: 窗口聚合频率，默认15min
        n_jobs: 并行计算的核数，默认4（单核环境可设为1）
        disable_progressbar: 是否禁用进度条，脚本运行建议设为True
        chunksize: tsfresh分块处理大小，大数据时设置（如10000）避免内存溢出
        rolling_window_sizes: 滚动窗口大小列表，如[5,10,20]，根据数据频率调整
    
    返回:
        提取后的特征DataFrame，索引为window_id（聚合窗口）
    """
    # === 前置校验 ===
    if 'openInterest' not in df_sec.columns:
        raise ValueError("输入DataFrame必须包含 'openInterest' 列")
    if not isinstance(df_sec.index, pd.DatetimeIndex):
        raise TypeError("输入DataFrame的索引必须是DatetimeIndex类型")
    
    # 清理索引时区 + 数据类型
    df_sec = df_sec.copy()
    df_sec.index = df_sec.index.tz_localize(None) if df_sec.index.tz is not None else df_sec.index
    df_sec['openInterest'] = pd.to_numeric(df_sec['openInterest'], errors='coerce')
    if df_sec['openInterest'].isnull().sum() / len(df_sec) > 0.1:
        logger.warning(f"openInterest列缺失值占比超过10%（{df_sec['openInterest'].isnull().sum()/len(df_sec):.2%}）")
    df_sec = df_sec.dropna(subset=['openInterest'])
    
    # === 新增：计算滚动窗口特征 ===
    logger.info(f"开始计算滚动窗口特征，窗口大小：{rolling_window_sizes}")
    df_rolling = df_sec[['openInterest']].copy()
    
    # 为每个窗口大小计算核心滚动特征
    for window in rolling_window_sizes:
        # 基础统计特征
        df_rolling[f'oi_roll{window}_mean'] = df_rolling['openInterest'].rolling(window=window, min_periods=2).mean()
        df_rolling[f'oi_roll{window}_std'] = df_rolling['openInterest'].rolling(window=window, min_periods=2).std()
        df_rolling[f'oi_roll{window}_max'] = df_rolling['openInterest'].rolling(window=window, min_periods=2).max()
        df_rolling[f'oi_roll{window}_min'] = df_rolling['openInterest'].rolling(window=window, min_periods=2).min()
        df_rolling[f'oi_roll{window}_range'] = df_rolling[f'oi_roll{window}_max'] - df_rolling[f'oi_roll{window}_min']
        
        # 趋势特征（滚动斜率）
        df_rolling[f'oi_roll{window}_slope'] = calculate_rolling_slope(df_rolling['openInterest'], window)
        
        # 动量特征（滚动收益率）
        df_rolling[f'oi_roll{window}_pct_change'] = df_rolling['openInterest'].pct_change(periods=window)
    
    # 填充滚动特征的NaN（前window-1个值用后续均值填充，避免丢失过多数据）
    df_rolling = df_rolling.fillna(method='bfill').fillna(method='ffill')
    
    # === 基础预处理（合并原始+滚动特征） ===
    df_prep = df_rolling.copy()
    df_prep['timestamp'] = df_prep.index
    df_prep['window_id'] = df_prep.index.floor(window_freq)
    
    # 过滤无效窗口（数据量<2）
    window_counts = df_prep.groupby('window_id').size()
    invalid_windows = window_counts[window_counts < 2].index
    if len(invalid_windows) > 0:
        logger.warning(f"过滤{len(invalid_windows)}个数据量<2的窗口")
        df_prep = df_prep[~df_prep['window_id'].isin(invalid_windows)]
    
    # === 扩展tsfresh特征配置：加入滚动特征的提取 ===
    # 1. 原始持仓量的特征配置（不变）
    base_features = {
        "maximum": None, "minimum": None, "mean": None, "median": None,
        "standard_deviation": None, "variance": None,
        "quantile": [{"q": 0.1}, {"q": 0.25}, {"q": 0.75}, {"q": 0.9}],
        "skewness": None, "kurtosis": None,
        "linear_trend": [{"attr": "slope"}, {"attr": "rvalue"}, {"attr": "intercept"}],
        "absolute_sum_of_changes": None, "mean_abs_change": None, "mean_change": None,
        "abs_energy": None, "autocorrelation": [{"lag": 1}, {"lag": 2}, {"lag": 5}, {"lag": 10}],
        "ratio_beyond_r_sigma": [{"r": 1}, {"r": 2}],
        "count_above_mean": None, "count_below_mean": None,
        "longest_strike_above_mean": None, "longest_strike_below_mean": None,
        "c3": [{"lag": 1}], "length": None
    }
    
    # 2. 为每个滚动特征添加基础统计（避免特征爆炸，只选核心）
    rolling_features = {}
    for window in rolling_window_sizes:
        for col_suffix in ['mean', 'std', 'slope']:  # 只选核心滚动特征做tsfresh提取
            col_name = f'oi_roll{window}_{col_suffix}'
            rolling_features[col_name] = {
                "mean": None, "std": None, "slope": None, "max": None, "min": None
            }
    
    # 合并配置：原始特征 + 滚动特征
    custom_settings = {}
    # 原始openInterest的特征
    custom_settings['openInterest'] = base_features
    # 滚动特征的简化特征
    custom_settings.update(rolling_features)
    
    # === 提取tsfresh特征（包含原始+滚动特征） ===
    try:
        logger.info(f"开始提取tsfresh特征（含滚动特征），窗口数：{df_prep['window_id'].nunique()}")
        # 调整extract_features的参数：指定所有要提取特征的列
        value_cols = ['openInterest'] + [f'oi_roll{w}_{s}' for w in rolling_window_sizes for s in ['mean', 'std', 'slope']]
        features = extract_features(
            df_prep,
            column_id='window_id',
            column_sort='timestamp',
            column_value=value_cols,  # 多列特征提取
            default_fc_parameters=custom_settings,
            n_jobs=n_jobs,
            disable_progressbar=disable_progressbar,
            chunksize=chunksize
        )
        features = impute(features)
    except Exception as e:
        logger.error(f"tsfresh特征提取失败：{str(e)}")
        raise e
    
    # === 清理列名 + 原有衍生特征 ===
    # 统一列名前缀，简化滚动特征列名
    features.columns = [
        col.replace('openInterest__', 'oi_')
           .replace('__mean', '_mean').replace('__std', '_std')
           .replace('__slope', '_slope').replace('__max', '_max').replace('__min', '_min')
        for col in features.columns
    ]
    # 将standard_deviation重命名为std（兼容原有逻辑）
    features.rename(columns={col: col.replace('standard_deviation', 'std') for col in features.columns}, inplace=True)
    
    # 手动添加OHLC特征 + 衍生特征（不变）
    grouper = df_prep.groupby('window_id')['openInterest']
    features['oi_first'] = grouper.first().reindex(features.index)
    features['oi_last'] = grouper.last().reindex(features.index)
    features['oi_net_change'] = features['oi_last'] - features['oi_first']
    features['oi_pct_change'] = features['oi_net_change'] / (features['oi_first'] + 1e-9)
    
    # 噪声比、变异系数、振幅比例（不变）
    if 'oi_absolute_sum_of_changes' in features.columns:
        features['oi_noise_ratio'] = features['oi_net_change'].abs() / (features['oi_absolute_sum_of_changes'] + 1e-9)
    else:
        features['oi_noise_ratio'] = np.nan
    if 'oi_mean' in features.columns and 'oi_std' in features.columns:
        features['oi_cv'] = features['oi_std'] / (features['oi_mean'] + 1e-9)
    else:
        features['oi_cv'] = np.nan
    if 'oi_maximum' in features.columns and 'oi_minimum' in features.columns:
        features['oi_range'] = features['oi_maximum'] - features['oi_minimum']
        features['oi_range_pct'] = features['oi_range'] / (features['oi_mean'] + 1e-9)
    else:
        features['oi_range'] = np.nan
        features['oi_range_pct'] = np.nan
    
    # === 异常值处理（不变） ===
    features = features.replace([np.inf, -np.inf], np.nan)
    for col in ['oi_noise_ratio', 'oi_cv', 'oi_range', 'oi_range_pct']:
        if col in features.columns:
            features[col] = features[col].fillna(features[col].median())
    
    logger.info(f"特征提取完成，最终特征数：{features.shape[1]}，样本数：{features.shape[0]}")
    return features