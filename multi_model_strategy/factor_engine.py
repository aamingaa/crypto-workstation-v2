"""
因子引擎模块
负责因子评估、标准化、相关性筛选
"""
import numpy as np
import pandas as pd
from gp_crypto_next.expressionProgram import FeatureEvaluator
from gp_crypto_next.functions import _function_map


class FactorEngine:
    """
    因子引擎：评估、标准化、筛选因子
    """
    
    def __init__(self, factor_expressions, X_all, feature_names, y_train, rolling_window=200):
        """
        Args:
            factor_expressions (list): 因子表达式列表
            X_all (np.ndarray): 全样本特征矩阵
            feature_names (list): 特征名称列表
            y_train (np.ndarray): 训练标签（用于相关性筛选）
            rolling_window (int): 滚动窗口大小
        """
        self.factor_expressions = factor_expressions
        self.X_all = X_all
        self.feature_names = feature_names
        self.y_train = y_train
        self.rolling_window = rolling_window
        
        self.evaluator = None
        self.factor_data = None  # pd.DataFrame
        self.valid_factor_expressions = []
        self.selected_factors = None
    
    def evaluate_expressions(self):
        """
        评估因子表达式，生成因子值矩阵
        
        Returns:
            self
        """
        print("正在评估因子表达式...")
        
        if not self.factor_expressions:
            raise ValueError("因子表达式列表为空")
        
        if self.X_all is None:
            raise ValueError("X_all 为空，请先加载数据")
        
        # 创建 FeatureEvaluator
        self.evaluator = FeatureEvaluator(_function_map, self.feature_names, self.X_all)
        
        # 评估每个因子表达式
        factor_values_list = []
        valid_expressions = []
        
        for i, expression in enumerate(self.factor_expressions):
            try:
                print(f"评估因子 {i+1}/{len(self.factor_expressions)}: {expression[:50]}...")
                factor_value = self.evaluator.evaluate(expression)
                factor_value = np.nan_to_num(factor_value)  # 处理NaN
                factor_values_list.append(factor_value)
                valid_expressions.append(expression)
            except Exception as e:
                print(f"  ⚠️  评估失败: {str(e)[:100]}")
                continue
        
        if not factor_values_list:
            raise ValueError("所有因子表达式评估失败")
        
        # 转换为DataFrame
        factor_columns = [f'gp_factor_{i}' for i in range(len(valid_expressions))]
        self.factor_data = pd.DataFrame(
            np.column_stack(factor_values_list),
            columns=factor_columns
        )
        
        self.valid_factor_expressions = valid_expressions
        
        print(f"成功评估 {len(valid_expressions)} 个因子")
        print(f"因子数据shape: {self.factor_data.shape}")
        
        return self
    
    def normalize(self, method='robust'):
        """
        因子标准化
        
        Args:
            method (str): 标准化方法
                - 'robust': log1p 压缩 + 完整 z-score（推荐）
                - 'zscore': 完整 z-score
                - 'simple': 简单除标准差
        
        Returns:
            self
        """
        print(f"正在进行因子标准化（方法: {method}）...")
        
        if method == 'robust':
            # log1p 压缩 + z-score
            arr = self.factor_data.values
            arr_compressed = np.sign(arr) * np.log1p(np.abs(arr))
            
            factors_data = pd.DataFrame(
                arr_compressed,
                columns=self.factor_data.columns,
                index=self.factor_data.index
            )
            factors_data = factors_data.replace([np.inf, -np.inf, np.nan], 0.0)
            
            factors_mean = factors_data.rolling(window=self.rolling_window, min_periods=1).mean()
            factors_std = factors_data.rolling(window=self.rolling_window, min_periods=1).std()
            factor_value = (factors_data - factors_mean) / factors_std
            factor_value = factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
            
            self.factor_data = factor_value
            print(f"✓ 使用 log1p + z-score 完成标准化（窗口={self.rolling_window}）")
            
        elif method == 'zscore':
            # 完整 z-score
            factors_mean = self.factor_data.rolling(window=self.rolling_window, min_periods=1).mean()
            factors_std = self.factor_data.rolling(window=self.rolling_window, min_periods=1).std()
            
            factor_value = (self.factor_data - factors_mean) / factors_std
            factor_value = factor_value.replace([np.nan, np.inf, -np.inf], 0.0)
            factor_value = factor_value.clip(-6, 6)
            
            self.factor_data = factor_value
            print(f"✓ 使用完整 z-score 完成标准化（窗口={self.rolling_window}）")
            
        elif method == 'simple':
            # 简单除标准差
            factors_std = self.factor_data.rolling(window=self.rolling_window, min_periods=1).std()
            factor_value = self.factor_data / factors_std
            factor_value = factor_value.replace([np.nan, np.inf, -np.inf], 0.0)
            factor_value = factor_value.clip(-6, 6)
            
            self.factor_data = factor_value
            print(f"✓ 使用简单方法完成标准化（窗口={self.rolling_window}）")
            
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
        
        print(f"  标准化后统计: mean={self.factor_data.mean().mean():.4f}, "
              f"std={self.factor_data.std().mean():.4f}, "
              f"min={self.factor_data.min().min():.4f}, "
              f"max={self.factor_data.max().max():.4f}")
        
        return self
    
    def select_by_correlation(self, corr_threshold=0.5):
        """
        基于相关性筛选因子（去除高度相关的因子）
        
        Args:
            corr_threshold (float): 相关性阈值
        
        Returns:
            self
        """
        print(f"正在进行因子去相关性筛选（阈值={corr_threshold}）...")
        
        fac_columns = list(self.factor_data.columns)
        X = self.factor_data[fac_columns]
        X_corr_matrix = X.corr()
        
        # 计算每个因子与收益的相关性
        factor_ret_corr = {}
        train_len = len(self.y_train)
        for col in fac_columns:
            corr = np.corrcoef(X[col].values[:train_len], self.y_train.flatten())[0, 1]
            factor_ret_corr[col] = abs(corr)
        
        # 去除高度相关的因子
        factor_list = fac_columns.copy()
        
        for i in range(len(fac_columns)):
            fct_1 = fac_columns[i]
            if fct_1 not in factor_list:
                continue
                
            for j in range(i+1, len(fac_columns)):
                fct_2 = fac_columns[j]
                if fct_2 not in factor_list:
                    continue
                    
                corr_value = X_corr_matrix.loc[fct_1, fct_2]
                
                if abs(corr_value) > corr_threshold:
                    # 保留与收益相关性更高的因子
                    if factor_ret_corr[fct_1] < factor_ret_corr[fct_2]:
                        if fct_1 in factor_list:
                            factor_list.remove(fct_1)
                            break
                    else:
                        if fct_2 in factor_list:
                            factor_list.remove(fct_2)
        
        self.selected_factors = factor_list
        print(f"去相关性筛选后剩余 {len(factor_list)} 个因子")
        
        return self
    
    def get_factor_data(self):
        """返回因子数据 DataFrame"""
        return self.factor_data
    
    def get_selected_factors(self):
        """返回筛选后的因子列表"""
        return self.selected_factors if self.selected_factors is not None else list(self.factor_data.columns)
