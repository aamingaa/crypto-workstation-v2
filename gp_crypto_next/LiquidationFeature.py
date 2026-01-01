import pandas as pd
import numpy as np
import talib
# 假设 norm, _safe_div 等工具函数在同目录的 utils 或本文件可访问
# 如果是在 originalFeature.py 定义的，建议挪到 utils.py 共享，或者传入
from .utils import norm, _safe_div  # 假设你已经把通用工具挪到了 utils

def get_advanced_liquidation_features(data_columns: list):
    """
    引用由 LiquidationFactorEngine 生成的特征列。
    
    参数:
    data_columns: 当前 DataFrame 的所有列名，用于自动匹配引擎产出的因子
    """
    features = {}
    
    # 定义匹配规则：LiquidationFactorEngine 产出的列通常带有特定前缀或特征
    # 例如：'feat_z_', 'feat_brk_', 'sum_', 'diff_', 'ratio_whale_' 等
    engine_prefixes = ['feat_z_', 'feat_brk_', 'sum_', 'count_', 'diff_', 'ratio_whale_', 'skew_']
    
    # 自动识别并注册
    for col in data_columns:
        if any(col.startswith(p) for p in engine_prefixes):
            # 使用闭包 capture 当前的 col 变量
            # 我们直接从 data 字典中提取已经预计算好的列
            features[col] = (lambda c=col: lambda data: norm(data.get(c, np.zeros(len(data['c'])))))()
            
    return features


