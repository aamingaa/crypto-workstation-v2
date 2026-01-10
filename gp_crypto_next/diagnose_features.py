"""
诊断特征数据质量的脚本
检查哪些特征容易产生大量相同值或标准差为0的情况
"""
import pandas as pd
import numpy as np
from .originalFeature import BaseFeature
from . import dataload

def diagnose_feature_quality(df, feature_names, window=100, n_segments=5):
    """
    诊断特征质量
    
    Args:
        df: 包含特征的DataFrame
        feature_names: 要检查的特征列表
        window: 滚动窗口大小
        n_segments: 切分的段数
    
    Returns:
        诊断结果DataFrame
    """
    results = []
    
    for feature in feature_names:
        if feature not in df.columns:
            print(f"Warning: {feature} not in dataframe")
            continue
            
        data = df[feature].values
        
        # 1. 检查整体统计量
        std_overall = np.std(data)
        mean_overall = np.mean(data)
        unique_ratio = len(np.unique(data)) / len(data)
        zero_ratio = np.sum(data == 0) / len(data)
        
        # 2. 检查滚动窗口内的标准差
        rolling_std = pd.Series(data).rolling(window=window).std()
        low_std_ratio = np.sum(rolling_std < 1e-6) / len(rolling_std)
        
        # 3. 切分成段，检查每段的标准差
        segments = np.array_split(data, n_segments)
        segments_with_zero_std = sum([1 for seg in segments if np.std(seg) < 1e-8])
        
        # 4. 检查连续相同值的最大长度
        max_consecutive_same = 1
        current_consecutive = 1
        for i in range(1, len(data)):
            if data[i] == data[i-1]:
                current_consecutive += 1
                max_consecutive_same = max(max_consecutive_same, current_consecutive)
            else:
                current_consecutive = 1
        
        results.append({
            'feature': feature,
            'std_overall': std_overall,
            'mean_overall': mean_overall,
            'unique_ratio': unique_ratio,
            'zero_ratio': zero_ratio,
            'low_std_ratio': low_std_ratio,
            'segments_zero_std': segments_with_zero_std,
            'max_consecutive_same': max_consecutive_same,
            'quality_score': (unique_ratio * 0.3 + 
                            (1 - zero_ratio) * 0.3 + 
                            (1 - low_std_ratio) * 0.2 +
                            (1 - segments_with_zero_std / n_segments) * 0.2)
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('quality_score', ascending=True)
    
    return results_df


def main():
    """主函数：加载数据并诊断 momentum 特征"""
    
    # 你需要根据实际情况修改这些参数
    print("加载数据...")
    # 示例：加载数据的代码（根据你的实际情况修改）
    # loader = dataload.DataLoader(...)
    # df = loader.load()
    
    # 临时示例：如果你有已经保存的特征数据
    # df = pd.read_csv('your_feature_data.csv')
    
    # Momentum 特征列表
    momentum_features = [
        'ori_ta_macd', 'close_macd', 'c_ta_tsf_5', 'h_ta_lr_angle_10', 
        'o_ta_lr_slope_10', 'v_trix_8_obv', 'ori_trix_8', 'ori_trix_21', 
        'ori_trix_55', 'obv_lr_slope_20', 'trend_slope_24', 
        'trend_slope_72', 'trend_slope_168', 'up_ratio_24',
        'donchian_pos_50', 'donchian_pos_200',
    ]
    
    print("开始诊断特征质量...")
    # results = diagnose_feature_quality(df, momentum_features)
    
    # print("\n特征质量诊断结果（按质量分数从低到高排序）：")
    # print("="*100)
    # print(results.to_string(index=False))
    
    # print("\n质量分数 < 0.5 的特征（建议移除）：")
    # bad_features = results[results['quality_score'] < 0.5]
    # print(bad_features[['feature', 'quality_score', 'zero_ratio', 'low_std_ratio']].to_string(index=False))
    
    print("\n提示：请根据你的实际数据路径修改此脚本后运行")


if __name__ == '__main__':
    main()

