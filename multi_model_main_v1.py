"""
多模型量化交易策略类（整合GP因子）
重构版本 - 面向对象设计

主要功能：
1. 从CSV文件读取GP生成的因子表达式
2. 使用eval方式解析和评估因子
3. 因子筛选和预处理
4. 多模型训练（OLS, Ridge, Lasso, XGBoost, LightGBM）
5. 回测和风险指标计算
6. 结果可视化

⚠️ 重要提示：与 gplearn 流程对比
==========================================
gplearn 的完整流程分为两个阶段：

【阶段1：GP 挖因子】（main_gp_new.py - run_genetic_programming, L268-277）
  - Label 选择：
    * IC 类指标：用 y_train（标准化 = ret_rolling_zscore）
    * Sharpe 类指标：用 ret_train（原始 = return_f）
  - 因子评估：只使用 np.nan_to_num(factor_values)，不标准化
  - Fitness：原始因子值 vs 相应 label

【阶段2：模型训练】（main_gp_new.py - go_model, L933）
  - model.fit(X_train, self.ret_train)  # 使用原始 label
  - 因子：挖掘出的因子的原始值（仅 np.nan_to_num）
  - 预测：model.predict(X) 直接作为仓位信号
  - 回测：pos × 实际价格收益率（不需要逆变换）

multi_model_main.py 的改进（修正后）：
  1. 加载数据：与 gplearn 一致 ✅
  2. 计算因子原始值：与 gplearn 一致 ✅
  3. **可选标准化因子**：
     - normalize_method=None（默认）：不标准化，与 gplearn 一致 ⭐推荐
     - normalize_method='simple/robust/zscore': 会改变因子值 ⚠️
  4. **训练模型**（改进点）：
     - use_normalized_label=True（默认）：用 y_train（标准化），训练更稳定 ⭐推荐
     - use_normalized_label=False: 用 ret_train（原始），与 go_model 一致
  5. **回测**：预测值作为仓位信号 × 实际价格收益率（不需要逆变换）✅

关键要点：
  - 因子：不标准化（normalize_method=None）
  - Label：推荐用标准化（use_normalized_label=True）- 训练更稳定，避免极值影响
  - 回测：预测值直接作为仓位信号，不需要 inverse_norm

为什么用标准化 label 训练？
  1. 数值稳定：避免极值样本影响训练
  2. 相对关系：模型学习的是因子与收益率的相对强弱关系
  3. 不需逆变换：预测值作为仓位信号，回测时乘以实际价格收益率即可

建议配置：
  strategy.run_full_pipeline(
      normalize_method=None,              # 不标准化因子
      # train_models 内部默认 use_normalized_label=True  # 使用标准化 label ⭐推荐
  )
"""

import sys
from pathlib import Path

# 参考 run_gp.py 的路径设置方式
project_root = Path(__file__).resolve().parent
# 确保本地项目根目录优先于环境同名第三方包
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# 添加 gp_crypto_next 包目录
pkg_dir = project_root / "gp_crypto_next"
if str(pkg_dir) not in sys.path:
    sys.path.insert(0, str(pkg_dir))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import math
import warnings
import os
import yaml
warnings.filterwarnings('ignore')

# 导入triple_barrier模块
from gp_crypto_next.triple_barrier import get_barrier, get_metalabel

# 设置matplotlib中文字体支持（Mac系统）
import platform

def setup_chinese_font_for_mac():
    """
    为Mac系统设置中文字体支持
    """
    if platform.system() == 'Darwin':  # Mac系统
        # 检查系统可用字体
        available_fonts = [f.name for f in mpl.font_manager.fontManager.ttflist]
        
        # Mac系统常见的中文字体列表（按优先级排序）
        mac_chinese_fonts = [
            'PingFang SC',      # 苹果默认中文字体
            'Songti SC',        # 宋体
            'STSong',          # 华文宋体
            'Arial Unicode MS', # 支持中文的Arial
            'SimHei',          # 黑体
            'Hiragino Sans GB', # 冬青黑体
            'STHeiti'          # 华文黑体
        ]
        
        # 寻找可用的中文字体
        selected_font = None
        for font in mac_chinese_fonts:
            if font in available_fonts:
                selected_font = font
                break
        
        if selected_font:
            plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 已设置中文字体: {selected_font}")
            return True
        else:
            plt.rcParams['axes.unicode_minus'] = False
            return False
    return True

# 调用字体设置函数
setup_chinese_font_for_mac()

# 导入GP相关模块
# 方式1：如果 gp_crypto_next 在 sys.path 中
# try:
#     from expressionProgram import FeatureEvaluator
#     from functions import _function_map
#     import dataload
# except ImportError:
#     # 方式2：使用完整路径
#     from gp_crypto_next.expressionProgram import FeatureEvaluator
#     from gp_crypto_next.functions import _function_map
#     import gp_crypto_next.dataload as dataload

from gp_crypto_next.expressionProgram import FeatureEvaluator
from gp_crypto_next.functions import _function_map
import gp_crypto_next.dataload as dataload

# 导入模型库
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from xgboost import XGBRegressor
import xgboost as xgb
import lightgbm as lgb
import joblib
import pickle

class QuantTradingStrategy:
    """
    多模型量化交易策略类（整合GP因子）
    """
    
    @classmethod
    def from_yaml_with_expressions(cls, yaml_path, factor_expressions, strategy_config=None):
        """
        从YAML配置文件和因子表达式列表创建策略实例
        
        Args:
            yaml_path (str): YAML配置文件路径
            factor_expressions (list): 因子表达式列表
            strategy_config (dict, optional): 策略配置
        
        Returns:
            QuantTradingStrategy: 策略实例
        """
        # 加载YAML配置
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        
        # 构建data_config（只保留必要参数）
        data_config = {
            'sym': yaml_config.get('sym', 'ETHUSDT'),
            'freq': yaml_config.get('freq', '15m'),
            'start_date_train': yaml_config.get('start_date_train'),
            'end_date_train': yaml_config.get('end_date_train'),
            'start_date_test': yaml_config.get('start_date_test'),
            'end_date_test': yaml_config.get('end_date_test'),
            'rolling_window': yaml_config.get('rolling_window', 2000),
            'data_dir': yaml_config.get('data_dir', ''),
            'read_frequency': yaml_config.get('read_frequency', 'monthly'),
            'data_source': yaml_config.get('data_source', 'kline'),
            # coarse_grain 相关（如果使用）
            'coarse_grain_period': yaml_config.get('coarse_grain_period', '2h'),
            'feature_lookback_bars': yaml_config.get('feature_lookback_bars', 8),
            'rolling_step': yaml_config.get('rolling_step', '15min'),
            'file_path': yaml_config.get('file_path', None),
            'timeframe': yaml_config.get('timeframe', None),
        }
        
        # 策略配置（使用合理默认值）
        config = {
            'return_period': yaml_config.get('y_train_ret_period', 1),
            'corr_threshold': 0.5,
            'fees_rate': 0.0005,
            'max_factors': 10,
            'model_save_path': './models',
            'annual_bars': 365 * 24 * 4
        }
        
        # 覆盖用户自定义配置
        if strategy_config:
            config.update(strategy_config)
        
        # 创建实例
        instance = cls(None, data_config, config)
        instance.factor_expressions = factor_expressions
        instance._use_expressions_mode = True
        
        print(f"已设置 {len(factor_expressions)} 个因子表达式")
        return instance
    
    @classmethod
    def from_expressions_simple(cls, factor_expressions, 
                                 sym='ETHUSDT', 
                                 train_dates=('2025-01-01', '2025-03-01'),
                                 test_dates=('2025-03-01', '2025-04-01'),
                                 **kwargs):
        """
        简化版本：直接从因子表达式创建策略（无需YAML文件）
        
        Args:
            factor_expressions (list): 因子表达式列表
            sym (str): 交易对，默认 'ETHUSDT'
            train_dates (tuple): 训练集日期范围 (开始, 结束)
            test_dates (tuple): 测试集日期范围 (开始, 结束)
            **kwargs: 其他配置参数
                - freq: 时间频率，默认 '15m'
                - rolling_window: 滚动窗口，默认 2000
                - max_factors: 最大因子数，默认 10
                - fees_rate: 手续费率，默认 0.0005
                - data_dir: 数据目录，默认 ''
        
        Returns:
            QuantTradingStrategy: 策略实例
        
        Example:
            >>> factor_expressions = [
            ...     'ta_rsi_14(close)',
            ...     'ta_ema_20(close)',
            ... ]
            >>> strategy = QuantTradingStrategy.from_expressions_simple(
            ...     factor_expressions,
            ...     sym='ETHUSDT',
            ...     train_dates=('2025-01-01', '2025-02-01'),
            ...     test_dates=('2025-02-01', '2025-03-01'),
            ...     max_factors=5
            ... )
            >>> strategy.run_full_pipeline()
        """
        # 数据配置
        data_config = {
            'sym': sym,
            'freq': kwargs.get('freq', '15m'),
            'start_date_train': train_dates[0],
            'end_date_train': train_dates[1],
            'start_date_test': test_dates[0],
            'end_date_test': test_dates[1],
            'rolling_window': kwargs.get('rolling_window', 2000),
            'data_dir': kwargs.get('data_dir', ''),
            'read_frequency': kwargs.get('read_frequency', 'monthly'),
            'data_source': kwargs.get('data_source', 'kline'),
            'coarse_grain_period': kwargs.get('coarse_grain_period', '2h'),
            'feature_lookback_bars': kwargs.get('feature_lookback_bars', 8),
            'rolling_step': kwargs.get('rolling_step', '15min'),
            'file_path': kwargs.get('file_path', None),
            'timeframe': kwargs.get('timeframe', None),
        }
        
        # 策略配置
        config = {
            'return_period': kwargs.get('return_period', 1),
            'corr_threshold': kwargs.get('corr_threshold', 0.5),
            'fees_rate': kwargs.get('fees_rate', 0.0005),
            'max_factors': kwargs.get('max_factors', 10),
            'model_save_path': kwargs.get('model_save_path', './models'),
        }
        
        # 创建实例
        instance = cls(None, data_config, config)
        instance.factor_expressions = factor_expressions
        instance._use_expressions_mode = True
        
        print(f"✓ 创建策略: {sym} | 训练集: {train_dates[0]}~{train_dates[1]} | 测试集: {test_dates[0]}~{test_dates[1]}")
        print(f"✓ 因子数量: {len(factor_expressions)}")
        
        return instance
    
    @classmethod
    def from_yaml(cls, yaml_path, factor_csv_path=None, strategy_config=None):
        """
        从YAML配置文件创建策略实例
        
        Args:
            yaml_path (str): YAML配置文件路径
            factor_csv_path (str, optional): GP因子CSV文件路径，如果不提供则自动推断
            strategy_config (dict, optional): 策略配置，会覆盖默认配置
        
        Returns:
            QuantTradingStrategy: 策略实例
        """
        # 加载YAML配置
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        
        # 构建data_config
        data_config = {
            'sym': yaml_config.get('sym', 'ETHUSDT'),
            'freq': yaml_config.get('freq', '15m'),
            'start_date_train': yaml_config.get('start_date_train'),
            'end_date_train': yaml_config.get('end_date_train'),
            'start_date_test': yaml_config.get('start_date_test'),
            'end_date_test': yaml_config.get('end_date_test'),
            'rolling_window': yaml_config.get('rolling_window', 2000),
            'data_dir': yaml_config.get('data_dir', ''),
            'read_frequency': yaml_config.get('read_frequency', 'monthly'),
            'timeframe': yaml_config.get('timeframe', None),
            'data_source': yaml_config.get('data_source', 'kline'),
            # coarse_grain 相关参数
            'coarse_grain_period': yaml_config.get('coarse_grain_period', '2h'),
            'feature_lookback_bars': yaml_config.get('feature_lookback_bars', 8),
            'rolling_step': yaml_config.get('rolling_step', '15min'),
            'file_path': yaml_config.get('file_path', None),
        }
        
        # 如果没有提供factor_csv_path，自动推断
        if factor_csv_path is None:
            factor_csv_path = (f"{data_config['sym']}_{data_config['freq']}_"
                             f"{yaml_config.get('y_train_ret_period', 1)}_"
                             f"{data_config['start_date_train']}_{data_config['end_date_train']}_"
                             f"{data_config['start_date_test']}_{data_config['end_date_test']}.csv.gz")
        
        # 合并策略配置
        config = {
            'return_period': yaml_config.get('y_train_ret_period', 1),
            'corr_threshold': 0.5,
            'position_size': 1.0,
            'clip_num': 5.0,
            'fixed_return': 0.0,
            'fees_rate': 0.0005,
            'annual_bars': 365 * 24 * 4,  # 默认15分钟
            'model_save_path': './models',
            'max_factors': 30,
        }
        
        # 如果提供了额外的策略配置，覆盖默认值
        if strategy_config:
            config.update(strategy_config)
        
        return cls(factor_csv_path, data_config, config)
    
    def __init__(self, factor_csv_path, data_config, config=None):
        """
        初始化策略
        
        Args:
            factor_csv_path (str or None): GP生成的因子CSV文件路径，如果为None则需要手动设置因子表达式
            data_config (dict): 数据加载配置（sym, freq, dates等）
            config (dict): 策略配置参数
        """
        self.factor_csv_path = factor_csv_path
        self.data_config = data_config
        self.config = config or self._get_default_config()
        
        # 数据相关
        self.raw_data = None
        self.factor_expressions = []  # 存储因子表达式
        self.factor_data = None  # 存储解析后的因子值
        self.feed_data = None
        
        # GP相关
        self.X_all = None
        self.feature_names = None
        self.evaluator = None
        
        # 训练数据
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.ret_train = None
        self.ret_test = None
        self.train_set_end_index = None
        self.z_index = None  # 时间索引
        
        # 模型
        self.models = {}
        self.predictions = {}
        
        # 回测结果
        self.backtest_results = {}
        self.performance_metrics = None
        
        # 模式标记
        self._use_expressions_mode = False  # 是否使用表达式模式（而非CSV文件）
        
        print(f"策略初始化完成")
    
    def _get_default_config(self):
        """获取默认配置"""
        return {
            'return_period': 1,  # 收益率计算周期
            'corr_threshold': 0.3,  # 相关性筛选阈值
            'sharpe_threshold': 0.2,  # 夏普比率筛选阈值
            'position_size': 1.0,  # 仓位大小
            'clip_num': 2.0,  # 仓位限制
            'fixed_return': 0.0,  # 无风险收益率
            'fees_rate': 0.0005,  # 手续费率
            'annual_bars': 365 * 24 * 4,  # 年化bar数（15分钟）
            'model_save_path': './',
            'max_factors': 50,  # 最多使用的因子数量
        }
    
    def load_data_from_dataload(self):
        """使用dataload模块加载数据"""
        print("正在使用dataload模块加载数据...")
        
        # 从data_config中获取参数
        sym = self.data_config['sym']
        freq = self.data_config['freq']
        start_date_train = self.data_config['start_date_train']
        end_date_train = self.data_config['end_date_train']
        start_date_test = self.data_config['start_date_test']
        end_date_test = self.data_config['end_date_test']
        rolling_w = self.data_config.get('rolling_window', 2000)
        data_dir = self.data_config.get('data_dir', '')
        read_frequency = self.data_config.get('read_frequency', 'monthly')
        timeframe = self.data_config.get('timeframe', None)
        
        # 根据data_source选择不同的加载方式
        data_source = self.data_config.get('data_source', 'kline')
        
        try:
            if str(data_source).lower() == 'coarse_grain':
                # 使用粗粒度特征加载
                print(f"使用粗粒度特征方法 (coarse_grain)")
                coarse_grain_period = self.data_config.get('coarse_grain_period', '2h')
                feature_lookback_bars = self.data_config.get('feature_lookback_bars', 8)
                rolling_step = self.data_config.get('rolling_step', '15min')
                file_path = self.data_config.get('file_path', None)
                
                # (self.X_all, X_train, self.y_train, self.ret_train, 
                #  X_test, self.y_test, self.ret_test, self.feature_names,
                #  open_train, open_test, close_train, close_test, 
                #  self.z_index, ohlc, self.y_p_train_origin,
                #  self.y_p_test_origin
                #  ) = dataload.data_prepare_coarse_grain_rolling(
                #     sym, freq, start_date_train, end_date_train,
                #     start_date_test, end_date_test,
                #     coarse_grain_period=coarse_grain_period,
                #     feature_lookback_bars=feature_lookback_bars,
                #     rolling_step=rolling_step,
                #     y_train_ret_period=self.config['return_period'],
                #     rolling_w=rolling_w,
                #     output_format='ndarry',
                #     data_dir=data_dir,
                #     read_frequency=read_frequency,
                #     timeframe=timeframe,
                #     file_path=file_path
                # )
                self.X_all, self.X_train, self.y_train, self.ret_train, self.X_test, self.y_test, self.ret_test, self.feature_names,self.open_train,self.open_test,self.close_train,self.close_test, self.z_index ,self.ohlc, self.y_p_train_origin, self.y_p_test_origin= dataload.data_prepare_coarse_grain_rolling(
                    sym, freq, start_date_train, end_date_train,
                    start_date_test, end_date_test,
                    coarse_grain_period=coarse_grain_period,
                    feature_lookback_bars=feature_lookback_bars,
                    rolling_step=rolling_step,
                    y_train_ret_period=self.config['return_period'],
                    rolling_w=rolling_w, 
                    output_format='ndarry',
                    data_dir=data_dir, 
                    read_frequency=read_frequency, 
                    timeframe=timeframe,
                    file_path=file_path,
                    include_categories = getattr(self, 'include_categories', ['momentum'])
                )
                
            elif str(data_source).lower() == 'kline':
                # 使用标准data_prepare
                (self.X_all, X_train, self.y_train, self.ret_train, 
                 X_test, self.y_test, self.ret_test, self.feature_names,
                 open_train, open_test, close_train, close_test, 
                 self.z_index, ohlc) = dataload.data_prepare(
                    sym, freq, start_date_train, end_date_train,
                    start_date_test, end_date_test, 
                    y_train_ret_period=self.config['return_period'],
                    rolling_w=rolling_w, 
                    data_dir=data_dir, 
                    read_frequency=read_frequency, 
                    timeframe=timeframe
                )
            else:
                raise ValueError(f"不支持的data_source: {data_source}")
                
            print(f"数据加载完成")
            print(f"X_all shape: {self.X_all.shape}")
            print(f"特征数量: {len(self.feature_names)}")
            print(f"训练集大小: {len(self.y_train)}")
            print(f"测试集大小: {len(self.y_test)}")
            
            # 存储OHLC数据用于回测
            # self.ohlc = ohlc
            # self.close_train = close_train
            # self.close_test = close_test
            # self.open_train = open_train
            # self.open_test = open_test
            
            return self
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            raise
    
    def set_factor_expressions(self, factor_expressions):
        """
        直接设置因子表达式列表（手动模式）
        
        Args:
            factor_expressions (list): 因子表达式列表
        
        Returns:
            self: 返回自身以支持链式调用
        """
        if not isinstance(factor_expressions, list):
            raise ValueError("factor_expressions 必须是列表类型")
        
        self.factor_expressions = factor_expressions
        self._use_expressions_mode = True
        
        print(f"已设置 {len(factor_expressions)} 个因子表达式")
        if factor_expressions:
            example = factor_expressions[0]
            print(f"示例因子: {example[:80] + '...' if len(example) > 80 else example}")
        
        return self
    
    def load_factor_expressions(self):
        """从CSV文件加载GP生成的因子表达式"""
        print(f"正在从 {self.factor_csv_path} 加载因子表达式...")
        
        try:
            # 读取CSV文件
            factor_df = pd.read_csv(self.factor_csv_path, compression='gzip' if self.factor_csv_path.endswith('.gz') else None)
            
            if 'expression' not in factor_df.columns:
                raise ValueError("CSV文件中未找到'expression'列")
            
            # 获取因子表达式列表
            all_expressions = factor_df['expression'].dropna().unique().tolist()
            
            print(f"共找到 {len(all_expressions)} 个唯一因子表达式")
            
            # 如果有性能指标，可以根据性能筛选因子
            # if 'fitness_sharpe_fixed_threshold_test' in factor_df.columns:
            if 'fitness_sharpe_test' in factor_df.columns:
                print("检测到夏普比率指标，根据测试集表现筛选因子...")
                # 按测试集夏普比率排序
                factor_df_sorted = factor_df.sort_values('fitness_sharpe_test', ascending=False)
                # 取前N个因子
                max_factors = min(self.config['max_factors'], len(factor_df_sorted))
                self.factor_expressions = factor_df_sorted['expression'].head(max_factors).tolist()
                print(f"筛选出表现最好的 {len(self.factor_expressions)} 个因子")
            else:
                # 没有性能指标，直接使用前N个
                max_factors = min(self.config['max_factors'], len(all_expressions))
                self.factor_expressions = all_expressions[:max_factors]
                print(f"使用前 {len(self.factor_expressions)} 个因子")
            
            return self
            
        except Exception as e:
            print(f"加载因子表达式失败: {e}")
            raise
    
    def evaluate_factor_expressions(self):
        """使用FeatureEvaluator解析和评估因子表达式"""
        print("正在评估因子表达式...")
        
        if not self.factor_expressions:
            raise ValueError("未找到因子表达式，请先运行 load_factor_expressions()")
        
        if self.X_all is None:
            raise ValueError("未加载数据，请先运行 load_data_from_dataload()")
        
        # 创建FeatureEvaluator
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
        
        # 存储有效的因子表达式
        self.valid_factor_expressions = valid_expressions
        
        print(f"成功评估 {len(valid_expressions)} 个因子")
        print(f"因子数据shape: {self.factor_data.shape}")
        
        return self
    
    def normalize_factors(self, method='robust'):
        """
        因子标准化
        
        Args:
            method (str): 标准化方法
                - 'robust': 使用 log1p 压缩 + 完整 z-score（推荐，与 gplearn 一致）
                - 'zscore': 完整 z-score（减均值 + 除标准差）
                - 'simple': 简单除标准差 + clip（计算快但对极端值敏感）
        
        Returns:
            self: 返回自身以支持链式调用
        """
        print(f"正在进行因子标准化（方法: {method}）...")
        
        rolling_w = self.data_config.get('rolling_window', 200)
        
        if method == 'robust':
            # 方法1: log1p 压缩 + 完整 z-score（最稳健）
            # Step 1: 对称 log1p 压缩（保留符号，压缩极端值）
            arr = self.factor_data.values
            arr_compressed = np.sign(arr) * np.log1p(np.abs(arr))
            
            # Step 2: 重建 DataFrame（保留列名和索引）
            factors_data = pd.DataFrame(
                arr_compressed, 
                columns=self.factor_data.columns,
                index=self.factor_data.index
            )
            factors_data = factors_data.replace([np.inf, -np.inf, np.nan], 0.0)
            
            # Step 3: 滚动 z-score 标准化（每列独立处理）
            factors_mean = factors_data.rolling(window=rolling_w, min_periods=1).mean()
            factors_std = factors_data.rolling(window=rolling_w, min_periods=1).std()
            factor_value = (factors_data - factors_mean) / factors_std
            
            # Step 4: 清理异常值
            factor_value = factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
            
            self.factor_data = factor_value
            print(f"✓ 使用 log1p + z-score 完成标准化（窗口={rolling_w}）")
            
        elif method == 'zscore':
            # 方法2: 完整 z-score（不做 log1p 压缩）
            factors_mean = self.factor_data.rolling(window=rolling_w, min_periods=1).mean()
            factors_std = self.factor_data.rolling(window=rolling_w, min_periods=1).std()
            
            factor_value = (self.factor_data - factors_mean) / factors_std
            factor_value = factor_value.replace([np.nan, np.inf, -np.inf], 0.0)
            factor_value = factor_value.clip(-6, 6)  # 异常值截断
            
            self.factor_data = factor_value
            print(f"✓ 使用完整 z-score 完成标准化（窗口={rolling_w}）")
            
        elif method == 'simple':
            # 方法3: 简单除标准差（不减均值，不中心化）
            factors_std = self.factor_data.rolling(window=rolling_w, min_periods=1).std()
            factor_value = self.factor_data / factors_std
            factor_value = factor_value.replace([np.nan, np.inf, -np.inf], 0.0)
            factor_value = factor_value.clip(-6, 6)
            
            self.factor_data = factor_value
            print(f"✓ 使用简单方法完成标准化（窗口={rolling_w}）")
            
        else:
            raise ValueError(f"不支持的标准化方法: {method}. 请选择 'robust', 'zscore' 或 'simple'")
        
        # 显示标准化后的统计信息
        print(f"  标准化后统计: mean={self.factor_data.mean().mean():.4f}, "
              f"std={self.factor_data.std().mean():.4f}, "
              f"min={self.factor_data.min().min():.4f}, "
              f"max={self.factor_data.max().max():.4f}")
        
        return self
    
    def factor_selection_by_correlation(self, corr_threshold):
        """基于相关性筛选因子"""
        print("正在进行因子去相关性筛选...")
        
        # 获取因子列名
        fac_columns = list(self.factor_data.columns)
        
        # 计算因子间相关性矩阵
        X = self.factor_data[fac_columns]
        X_corr_matrix = X.corr()
        
        # 计算每个因子与收益的相关性
        factor_ret_corr = {}
        for col in fac_columns:
            # 使用训练集数据计算相关性
            train_len = len(self.y_train)
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
        
        print(f"去相关性筛选后剩余 {len(factor_list)} 个因子")
        return factor_list
    
    def select_factors(self):
        """因子筛选"""
        print("正在进行因子筛选...")
        
        # 基于相关性筛选
        selected_factors = self.factor_selection_by_correlation(
            self.config['corr_threshold'])
        
        self.selected_factors = selected_factors
        
        return self
    
    def generate_triple_barrier_labels(self, pt_sl=[2, 2], max_holding=[0, 4], side_prediction=None):
        """
        生成 Triple Barrier 标签
        
        Args:
            pt_sl (list): [profit_taking_multiplier, stop_loss_multiplier]
                         例如 [2, 2] 表示止盈和止损都是目标波动率的2倍
            max_holding (list): [days, hours] 最大持仓时间
                               例如 [0, 4] 表示最多持有4小时
            side_prediction (pd.Series, optional): 预测方向，1=做多，-1=做空
                                                   如果为None，默认都是做多
        
        Returns:
            self
        """
        print("正在生成 Triple Barrier 标签...")
        
        # 准备价格序列（需要有时间索引）
        if self.z_index is None or self.ohlc is None:
            raise ValueError("需要先加载数据（包含时间索引和OHLC数据）")
        
        # 创建带时间索引的收盘价序列
        close_series = pd.Series(
            data=self.ohlc[:, 3],  # close price
            index=pd.to_datetime(self.z_index)
        )
        
        # 计算目标波动率（使用滚动标准差）
        rolling_window = self.data_config.get('rolling_window', 2000)
        target_volatility = close_series.pct_change().rolling(
            window=min(rolling_window, len(close_series)//2)
        ).std()
        target_volatility = target_volatility.fillna(method='bfill')
        
        # 生成入场点（所有时间点）
        enter_points = close_series.index
        
        # 如果有方向预测，使用预测方向；否则默认做多
        if side_prediction is None:
            side_series = pd.Series(1.0, index=enter_points)
        else:
            side_series = side_prediction
        
        # 调用 triple_barrier
        barrier_results = get_barrier(
            close=close_series,
            enter=enter_points,
            pt_sl=pt_sl,
            max_holding=max_holding,
            target=target_volatility,
            side=side_series
        )
        
        # 生成 meta-label（1=盈利，0=亏损）
        meta_labels = get_metalabel(barrier_results)
        
        # 存储结果
        self.barrier_results = barrier_results
        self.meta_labels = meta_labels
        
        print(f"Triple Barrier 标签生成完成")
        print(f"总交易次数: {len(barrier_results)}")
        print(f"盈利交易: {(meta_labels == 1).sum()} ({(meta_labels == 1).sum()/len(meta_labels):.2%})")
        print(f"亏损交易: {(meta_labels == 0).sum()} ({(meta_labels == 0).sum()/len(meta_labels):.2%})")
        print(f"平均收益: {barrier_results['ret'].mean():.4f}")
        print(f"收益标准差: {barrier_results['ret'].std():.4f}")
        
        return self
    
    def use_triple_barrier_as_y(self):
        """
        使用 Triple Barrier 的收益作为训练目标（替代原来的固定周期收益）
        
        注意：这会改变 y_train 和 y_test
        """
        print("正在使用 Triple Barrier 收益作为训练目标...")
        
        if not hasattr(self, 'barrier_results'):
            raise ValueError("请先运行 generate_triple_barrier_labels()")
        
        # 将 barrier 收益对齐到原始索引
        barrier_ret = self.barrier_results['ret'].values
        
        # 更新训练目标
        train_len = len(self.y_train)
        self.y_train = barrier_ret[:train_len].reshape(-1, 1)
        self.y_test = barrier_ret[train_len:len(self.y_train)+len(self.y_test)].reshape(-1, 1)
        
        # 同时更新 ret_train 和 ret_test
        self.ret_train = self.y_train.flatten()
        self.ret_test = self.y_test.flatten()
        
        print(f"已替换训练目标为 Triple Barrier 收益")
        print(f"训练集收益范围: [{self.ret_train.min():.4f}, {self.ret_train.max():.4f}]")
        print(f"测试集收益范围: [{self.ret_test.min():.4f}, {self.ret_test.max():.4f}]")
        
        return self
    
    def prepare_training_data(self):
        """准备训练数据"""
        print("正在准备训练数据...")
        
        # 训练集和测试集已经在load_data中划分好了
        train_len = len(self.y_train)
        
        # 如果没有进行因子选择，使用所有因子
        if not hasattr(self, 'selected_factors') or self.selected_factors is None:
            print("  未进行因子选择，使用所有因子")
            self.selected_factors = self.factor_data.columns.tolist()
        
        print(f"  使用因子数量: {len(self.selected_factors)}")
        
        self.X_train = self.factor_data[self.selected_factors].iloc[:train_len].values
        self.X_test = self.factor_data[self.selected_factors].iloc[train_len:].values
        
        # y_train 和 y_test 已在 load_data 中设置
        # 但需要确保形状正确
        if len(self.y_train.shape) == 1:
            self.y_train = self.y_train.reshape(-1, 1)
        if len(self.y_test.shape) == 1:
            self.y_test = self.y_test.reshape(-1, 1)
        
        print(f"训练集大小: X={self.X_train.shape}, y={self.y_train.shape}")
        print(f"测试集大小: X={self.X_test.shape}, y={self.y_test.shape}")
        
        return self
    
    def train_models(self, use_normalized_label=True):
        """
        训练多个模型 - 参考analyzer.go_model的逻辑
        
        Args:
            use_normalized_label (bool): 是否使用标准化后的 label
                - True: 使用 y_train（ret_rolling_zscore，标准化label）⭐推荐
                - False: 使用 ret_train（原始 return_f）
        
        重要说明：
            为什么用标准化 label？
            1. 数值稳定：避免极值影响训练
            2. 相对关系：模型学习因子与收益率的相对关系
            3. 回测时不需要逆变换：预测值直接作为仓位信号，乘以实际价格收益率
            
            gplearn 的 go_model 可能用 ret_train（原始），但用标准化 label 训练效果更好
        """
        print("正在训练模型...")
        
        # 选择训练标签
        if use_normalized_label:
            print("ℹ️  使用标准化 label (y_train = ret_rolling_zscore) ⭐推荐")
            train_label = self.y_train
            test_label = self.y_test
        else:
            print("⚠️  使用原始 label (ret_train = return_f)")
            train_label = self.ret_train.reshape(-1, 1) if len(self.ret_train.shape) == 1 else self.ret_train
            test_label = self.ret_test.reshape(-1, 1) if len(self.ret_test.shape) == 1 else self.ret_test
        
        # 1. 线性回归（主模型 - 类似go_model）
        print("训练线性回归模型...")
        lr_model = LinearRegression(fit_intercept=True)
        lr_model.fit(self.X_train, train_label)
        self.models['LinearRegression'] = lr_model
        print(f"  系数: {lr_model.coef_.flatten()[:5]}... (显示前5个)")
        print(f"  截距: {lr_model.intercept_}")
        
        # 2. Ridge回归
        print("训练Ridge回归模型...")
        ridge_model = Ridge(alpha=0.2, fit_intercept=True)
        ridge_model.fit(self.X_train, train_label)
        self.models['Ridge'] = ridge_model
        
        # 3. Lasso回归
        print("训练Lasso回归模型...")
        lasso_model = LassoCV(fit_intercept=True, max_iter=5000)
        lasso_model.fit(self.X_train, train_label.flatten())
        self.models['Lasso'] = lasso_model
        
        # 4. XGBoost
        print("训练XGBoost模型...")
        X_train_df = pd.DataFrame(self.X_train, columns=self.selected_factors)
        y_train_series = pd.Series(train_label.flatten())
        X_test_df = pd.DataFrame(self.X_test, columns=self.selected_factors)
        y_test_series = pd.Series(test_label.flatten())
        
        xgb_model = XGBRegressor(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=100,
            objective='reg:squarederror',
            random_state=0,
            early_stopping_rounds=20
        )
        
        xgb_model.fit(
            X_train_df, y_train_series,
            eval_set=[(X_test_df, y_test_series)],
            verbose=False
        )
        
        self.models['XGBoost'] = xgb_model
        
        # 5. LightGBM
        print("训练LightGBM模型...")
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting': 'gbdt',
            'learning_rate': 0.054,
            'max_depth': 3,
            'num_leaves': 32,
            'min_data_in_leaf': 50,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.5,
            'lambda_l1': 0.05,
            'lambda_l2': 120,
            'verbose': -1
        }
        
        lgb_train = lgb.Dataset(X_train_df, y_train_series)
        lgb_val = lgb.Dataset(X_test_df, y_test_series, reference=lgb_train)
        
        lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=500,
            valid_sets=lgb_val,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        self.models['LightGBM'] = lgb_model
        
        print("所有模型训练完成")
        return self
    
    def make_predictions(self, weight_method='equal'):
        """
        生成预测 - 支持多模型集成
        
        Args:
            weight_method (str): 权重计算方法，可选 'equal'（等权重）或 'sharpe'（基于夏普比率）
        """
        print("正在生成预测...")
        
        # 存储所有模型的预测结果用于集成
        all_train_predictions = []
        all_test_predictions = []
        model_names = []
        
        for model_name, model in self.models.items():
            if model_name == 'LightGBM':
                train_pred = model.predict(self.X_train)
                test_pred = model.predict(self.X_test)
            else:
                train_pred = model.predict(self.X_train).flatten()
                test_pred = model.predict(self.X_test).flatten()
            
            # 根据历史分位数映射到仓位
            # 参考go_model中的逻辑
            min_val = abs(np.percentile(train_pred, 99))
            max_val = abs(np.percentile(train_pred, 1))
            scale_n = 2 / (min_val + max_val) if (min_val + max_val) > 0 else 1.0
            
            # 缩放并裁剪到[-5, 5]
            train_pred_scaled = (train_pred * scale_n).clip(-5, 5)
            test_pred_scaled = (test_pred * scale_n).clip(-5, 5)
            
            # 存储单个模型的预测结果
            self.predictions[model_name] = {
                'train': train_pred_scaled,
                'test': test_pred_scaled
            }
            
            # 收集用于集成
            all_train_predictions.append(train_pred_scaled)
            all_test_predictions.append(test_pred_scaled)
            model_names.append(model_name)
        
        # 计算组合权重
        if weight_method == 'equal':
            # 等权重组合
            weights = {name: 1.0/len(model_names) for name in model_names}
            print(f"使用等权重组合方式")
        elif weight_method == 'sharpe':
            # 基于夏普比率的权重
            print(f"正在计算基于夏普比率的权重...")
            sharpe_ratios = {}
            
            for model_name in model_names:
                # 快速计算训练集的夏普比率
                train_pos = self.predictions[model_name]['train']
                _, train_metrics = self.real_trading_simulator(train_pos, 'train', self.config['fees_rate'])
                
                # 使用训练集夏普比率的绝对值作为权重
                sharpe_ratios[model_name] = abs(train_metrics['Sharpe Ratio'])
                print(f"  {model_name}: Sharpe = {train_metrics['Sharpe Ratio']:.4f}")
            
            # 计算权重（确保权重和为1）
            total_sharpe = sum(sharpe_ratios.values())
            if total_sharpe > 0:
                weights = {name: sharpe/total_sharpe for name, sharpe in sharpe_ratios.items()}
            else:
                # 如果所有夏普比率都为0，使用等权重
                print("  ⚠️  所有模型夏普比率均为0，改用等权重")
                weights = {name: 1.0/len(model_names) for name in model_names}
        else:
            raise ValueError(f"不支持的权重计算方法: {weight_method}")
        
        # 存储权重信息
        self.ensemble_weights = weights
        print("\n模型组合权重:")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.2%}")
        
        # 计算加权组合预测
        ensemble_train_pred = np.zeros_like(all_train_predictions[0])
        ensemble_test_pred = np.zeros_like(all_test_predictions[0])
        
        for i, model_name in enumerate(model_names):
            ensemble_train_pred += weights[model_name] * all_train_predictions[i]
            ensemble_test_pred += weights[model_name] * all_test_predictions[i]
        
        # 添加组合模型预测结果
        self.predictions['Ensemble'] = {
            'train': ensemble_train_pred,
            'test': ensemble_test_pred
        }
        
        print(f"\n预测生成完成，共 {len(self.predictions)} 个模型（包含Ensemble）")
        return self
    
    def real_trading_simulator(self, pos, data_range='test', fee=0.0005):
        """
        模拟真实交易场景 - 参考analyzer.real_trading_simulator
        
        Args:
            pos: 仓位序列
            data_range: 'train' 或 'test'
            fee: 手续费率
        """
        # 获取对应的开盘价和收盘价
        if data_range == 'train':
            open_data = self.open_train
            close_data = self.close_train
        elif data_range == 'test':
            open_data = self.open_test
            close_data = self.close_test
        else:
            raise ValueError(f"不支持的data_range: {data_range}")
        
        # 转换为 numpy 数组并确保正确的形状
        if isinstance(open_data, pd.Series):
            open_data = open_data.values
        if isinstance(close_data, pd.Series):
            close_data = close_data.values
        
        # 确保pos是1维数组
        pos = np.asarray(pos).flatten()
        
        # 确保长度匹配
        min_len = min(len(pos), len(open_data), len(close_data))
        pos = pos[:min_len]
        open_data = open_data[:min_len]
        close_data = close_data[:min_len]
        
        next_open = np.concatenate((open_data[1:], np.array([close_data[-1]])))
        close = close_data
        
        real_pos = pos
        pos_change = np.concatenate((np.array([0]), np.diff(real_pos)))
        
        # 决定交易价格
        which_price_to_trade = np.where(
            pos_change > 0,
            np.maximum(close, next_open),  # 买入用更高价格
            np.where(
                pos_change < 0,
                np.minimum(close, next_open),  # 卖出用更低价格
                close
            )
        )
        
        next_trade_close = np.concatenate((which_price_to_trade[1:], np.array([which_price_to_trade[-1]])))
        rets = np.log(next_trade_close) - np.log(which_price_to_trade)
        
        # 计算收益（扣除手续费）
        gain_loss = real_pos * rets - abs(pos_change) * fee
        pnl = gain_loss.cumsum()
        
        # 调试信息
        if np.any(np.isnan(gain_loss)) or np.any(np.isinf(gain_loss)):
            print(f"    ⚠️  警告: gain_loss中有NaN或Inf值")
            print(f"    real_pos stats: min={np.nanmin(real_pos):.4f}, max={np.nanmax(real_pos):.4f}, nan_count={np.sum(np.isnan(real_pos))}")
            print(f"    rets stats: min={np.nanmin(rets):.4f}, max={np.nanmax(rets):.4f}, nan_count={np.sum(np.isnan(rets))}")
            print(f"    which_price stats: min={np.nanmin(which_price_to_trade):.4f}, max={np.nanmax(which_price_to_trade):.4f}")
        
        # 计算性能指标
        win_rate_bar = np.sum(gain_loss > 0) / len(gain_loss) if len(gain_loss) > 0 else 0
        avg_gain_bar = np.mean(gain_loss[gain_loss > 0]) if np.any(gain_loss > 0) else 0
        avg_loss_bar = np.abs(np.mean(gain_loss[gain_loss < 0])) if np.any(gain_loss < 0) else 0
        profit_loss_ratio_bar = avg_gain_bar / avg_loss_bar if avg_loss_bar != 0 else np.inf
        
        annual_bars = self.config['annual_bars']
        annual_return = np.mean(gain_loss) * annual_bars
        sharpe_ratio = annual_return / (np.std(gain_loss) * np.sqrt(annual_bars)) if np.std(gain_loss) > 0 else 0
        
        # 计算最大回撤
        peak_values = np.maximum.accumulate(pnl)
        # 避免初始值为0导致的问题
        peak_values = np.where(peak_values == 0, 1.0, peak_values)
        drawdowns = (pnl - peak_values) / np.abs(peak_values)
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # 计算卡尔玛比率（避免除以0）
        if max_drawdown < -0.0001:  # 有实际回撤
            Calmar_Ratio = annual_return / abs(max_drawdown)
        else:
            Calmar_Ratio = np.inf if annual_return > 0 else 0
        
        metrics = {
            "Win Rate": win_rate_bar,
            "Profit/Loss Ratio": profit_loss_ratio_bar,
            "Annual Return": annual_return,
            "MAX_Drawdown": max_drawdown,
            "Sharpe Ratio": sharpe_ratio,
            "Calmar Ratio": Calmar_Ratio
        }
        
        return pnl, metrics
    
    def backtest(self, model_name):
        """回测指定模型"""
        print(f"正在回测 {model_name} 模型...")
        
        if model_name not in self.predictions:
            raise ValueError(f"模型 {model_name} 的预测结果不存在")
        
        # 获取仓位
        train_pos = self.predictions[model_name]['train']
        test_pos = self.predictions[model_name]['test']
        
        # 调试信息
        print(f"  训练集仓位: shape={train_pos.shape}, range=[{train_pos.min():.2f}, {train_pos.max():.2f}]")
        print(f"  测试集仓位: shape={test_pos.shape}, range=[{test_pos.min():.2f}, {test_pos.max():.2f}]")
        print(f"  训练集价格数据长度: open={len(self.open_train)}, close={len(self.close_train)}")
        print(f"  测试集价格数据长度: open={len(self.open_test)}, close={len(self.close_test)}")
        
        # 回测
        train_pnl, train_metrics = self.real_trading_simulator(train_pos, 'train', self.config['fees_rate'])
        test_pnl, test_metrics = self.real_trading_simulator(test_pos, 'test', self.config['fees_rate'])
        
        self.backtest_results[model_name] = {
            'train_pnl': train_pnl,
            'test_pnl': test_pnl,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
        
        print(f"{model_name} 回测完成")
        print(f"样本内夏普比率: {train_metrics['Sharpe Ratio']:.4f}")
        print(f"样本外夏普比率: {test_metrics['Sharpe Ratio']:.4f}")
        
        return self
    
    def backtest_all_models(self):
        """回测所有模型"""
        print("正在回测所有模型...")
        
        for model_name in self.predictions.keys():
            self.backtest(model_name)
        
        return self
    
    def plot_results(self, model_name='Ensemble'):
        """
        绘制回测结果
        
        Args:
            model_name (str): 模型名称，默认为 'Ensemble'（推荐）
        """
        if model_name not in self.backtest_results:
            print(f"模型 {model_name} 的回测结果不存在")
            return
        
        result = self.backtest_results[model_name]
        train_pnl = result['train_pnl']
        test_pnl = result['test_pnl']
        train_metrics = result['train_metrics']
        test_metrics = result['test_metrics']
        
        # 获取时间索引
        train_index = self.z_index[:len(train_pnl)]
        test_index = self.z_index[len(train_pnl):len(train_pnl) + len(test_pnl)]
        
        # 创建图表
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        
        # 根据模型类型设置标题
        if model_name == 'Ensemble':
            title = f'多模型组合 (Ensemble) 回测结果'
            color = 'red'
        else:
            title = f'{model_name} 回测结果'
            color = 'green'
        
        fig.suptitle(title, fontsize=18, fontweight='bold')
        
        # 训练集价格
        axs[0, 0].plot(train_index, self.close_train, 'b-', linewidth=1.5)
        axs[0, 0].set_title('训练集价格', fontsize=12, fontweight='bold')
        axs[0, 0].set_ylabel('价格', fontsize=10)
        axs[0, 0].grid(True, alpha=0.3, linestyle='--')
        
        # 训练集PnL
        axs[1, 0].plot(train_index, train_pnl, 'g-', linewidth=2)
        axs[1, 0].set_title('训练集累计PnL', fontsize=12, fontweight='bold')
        axs[1, 0].set_ylabel('累计收益', fontsize=10)
        axs[1, 0].grid(True, alpha=0.3, linestyle='--')
        axs[1, 0].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # 添加训练集指标
        train_text = (f"年化收益: {train_metrics['Annual Return']:.2%}\n"
                     f"夏普比率: {train_metrics['Sharpe Ratio']:.3f}\n"
                     f"最大回撤: {train_metrics['MAX_Drawdown']:.2%}")
        axs[1, 0].text(0.02, 0.98, train_text,
                      transform=axs[1, 0].transAxes,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6),
                      fontsize=9)
        
        # 测试集价格
        axs[0, 1].plot(test_index, self.close_test, 'b-', linewidth=1.5)
        axs[0, 1].set_title('测试集价格', fontsize=12, fontweight='bold')
        axs[0, 1].set_ylabel('价格', fontsize=10)
        axs[0, 1].grid(True, alpha=0.3, linestyle='--')
        
        # 测试集PnL
        axs[1, 1].plot(test_index, test_pnl, color=color, linewidth=2.5)
        axs[1, 1].set_title('测试集累计PnL', fontsize=12, fontweight='bold')
        axs[1, 1].set_ylabel('累计收益', fontsize=10)
        axs[1, 1].grid(True, alpha=0.3, linestyle='--')
        axs[1, 1].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # 添加测试集指标（如果是Ensemble，显示权重信息）
        if model_name == 'Ensemble' and hasattr(self, 'ensemble_weights'):
            weight_info = "\n".join([f"{name}: {w:.1%}" for name, w in self.ensemble_weights.items()])
            metrics_text = (f"年化收益: {test_metrics['Annual Return']:.2%}\n"
                          f"夏普比率: {test_metrics['Sharpe Ratio']:.3f}\n"
                          f"最大回撤: {test_metrics['MAX_Drawdown']:.2%}\n"
                          f"卡尔玛比率: {test_metrics['Calmar Ratio']:.3f}\n"
                          f"胜率: {test_metrics['Win Rate']:.2%}\n"
                          f"\n模型权重:\n{weight_info}")
        else:
            metrics_text = (f"年化收益: {test_metrics['Annual Return']:.2%}\n"
                          f"夏普比率: {test_metrics['Sharpe Ratio']:.3f}\n"
                          f"最大回撤: {test_metrics['MAX_Drawdown']:.2%}\n"
                          f"卡尔玛比率: {test_metrics['Calmar Ratio']:.3f}\n"
                          f"胜率: {test_metrics['Win Rate']:.2%}\n"
                          f"盈亏比: {test_metrics['Profit/Loss Ratio']:.3f}")
        
        axs[1, 1].text(0.02, 0.98, metrics_text,
                      transform=axs[1, 1].transAxes,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
                      fontsize=9)
        
        plt.tight_layout()
        plt.show(block=False)
        
        return self
    
    def get_performance_summary(self):
        """获取所有模型的绩效汇总"""
        if not self.backtest_results:
            print("请先运行回测")
            return None
        
        summary_data = []
        
        for model_name, results in self.backtest_results.items():
            train_metrics = results['train_metrics']
            test_metrics = results['test_metrics']
            
            summary_data.append({
                '模型': model_name,
                '样本内年化收益': train_metrics['Annual Return'],
                '样本内夏普比率': train_metrics['Sharpe Ratio'],
                '样本内最大回撤': train_metrics['MAX_Drawdown'],
                '样本外年化收益': test_metrics['Annual Return'],
                '样本外夏普比率': test_metrics['Sharpe Ratio'],
                '样本外最大回撤': test_metrics['MAX_Drawdown']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.round(4)
        
        print("\n模型绩效汇总:")
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    def save_models(self):
        """保存模型"""
        print("正在保存模型...")
        
        save_path = Path(self.config['model_save_path'])
        save_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            if model_name == 'LightGBM':
                model.save_model(str(save_path / 'lgb_model.txt'))
            else:
                joblib.dump(model, str(save_path / f'{model_name.lower()}_model.pkl'))
        
        # 保存因子表达式映射
        factor_mapping = {
            'selected_factors': self.selected_factors,
            'factor_expressions': dict(zip(self.selected_factors, 
                                          [self.valid_factor_expressions[int(f.split('_')[-1])] 
                                           for f in self.selected_factors]))
        }
        with open(save_path / 'factor_mapping.pkl', 'wb') as f:
            pickle.dump(factor_mapping, f)
        
        print(f"模型保存完成，路径: {save_path}")
        return self
    
    def run_full_pipeline(self, weight_method='equal', normalize_method=None, enable_factor_selection=False):
        """
        运行完整的策略流程
        
        Args:
            weight_method (str): 模型集成权重方法，'equal' 或 'sharpe'
            normalize_method (str or None): 因子标准化方法
                - None: 不标准化（默认，与 gplearn 一致）⭐推荐
                - 'simple': 简单除标准差（与 gplearn 的 norm 函数一致）
                - 'robust': log1p压缩 + 完整z-score
                - 'zscore': 完整z-score（中心化）
            enable_factor_selection (bool): 是否启用基于相关性的因子筛选（默认False，使用所有因子）
            
        重要说明：
            gplearn 挖因子时使用的是因子原始值（不标准化），所以 normalize_method=None 可以
            复现 gplearn 训练集的表现。如果标准化会导致因子值完全改变！
        """
        print("="*60)
        print("开始运行完整的量化策略流程（整合GP因子）")
        print("="*60)
        
        # 加载数据
        self.load_data_from_dataload()
        
        # 如果不是表达式模式，需要从CSV加载因子表达式
        if not self._use_expressions_mode:
            self.load_factor_expressions()
        else:
            print(f"使用预设的 {len(self.factor_expressions)} 个因子表达式")
        
        # 执行剩余流程
        pipeline = self.evaluate_factor_expressions()
        
        # 可选：因子标准化（默认不标准化，与 gplearn 一致）
        if normalize_method is not None:
            print(f"⚠️  警告: 启用因子标准化可能导致效果与 gplearn 不一致！")
            pipeline = pipeline.normalize_factors(method=normalize_method)
        else:
            print("ℹ️  不标准化因子（与 gplearn 一致）")
        
        # 可选：因子筛选（基于相关性）
        if enable_factor_selection:
            pipeline = pipeline.select_factors()
        
        # 继续后续流程
        (pipeline
         .prepare_training_data()
         .train_models(use_normalized_label=True)  # 使用标准化 label，训练更稳定 ⭐推荐
         .make_predictions(weight_method=weight_method)
         .backtest_all_models()
        )
        
        # 显示绩效汇总
        summary_df = self.get_performance_summary()
        
        print("\n" + "="*60)
        print("策略流程执行完成！")
        print("="*60)
        return self


# 使用示例
if __name__ == "__main__":
    # ========== 方式1: 从YAML配置文件创建（推荐） ==========
    # yaml_path = 'gp_crypto_next/coarse_grain_parameters.yaml'
    
    # # 自动推断因子CSV文件名，或手动指定
    # factor_csv_path = 'gp_models/ETHUSDT_15m_1_2025-01-01_2025-01-20_2025-01-20_2025-01-31.csv.gz'
    
    # # 可选：额外的策略配置（会覆盖默认值）
    # strategy_config = {
    #     'corr_threshold': 0.5,  # 因子去相关阈值
    #     'max_factors': 10,  # 最多使用30个因子
    #     'fees_rate': 0.0005,  # 手续费率
    #     'model_save_path': './models',
    # }
    
    # # 从YAML创建策略
    # strategy = QuantTradingStrategy.from_yaml(
    #     yaml_path=yaml_path,
    #     factor_csv_path=factor_csv_path,
    #     strategy_config=strategy_config
    # )
    
    # # 运行完整流程
    # # weight_method 可选: 'equal' (等权重) 或 'sharpe' (基于夏普比率)
    # strategy.run_full_pipeline(weight_method='equal')
    
    
    # ========== 方式2: 从YAML + 因子表达式列表创建 ==========
    yaml_path = 'gp_crypto_next/coarse_grain_parameters.yaml'
    
    # 定义因子表达式列表
    factor_expressions = [
        'ta_dema_55(ta_dx_25(neg(h), ta_trix_8(h), ta_trix_21(ori_trix_21)))',
        'ta_lr_slope_20(ts_delta_17(ori_ta_macd))',
    ]
    
    # 从YAML和因子表达式创建策略
    strategy = QuantTradingStrategy.from_yaml_with_expressions(
        yaml_path=yaml_path,
        factor_expressions=factor_expressions,
        strategy_config={'max_factors': 10}  # 可选配置
    )
    
    # 运行完整流程
    # ⭐ 关键: normalize_method=None 可以复现 gplearn 的效果！
    # normalize_method 可选:
    #   - None: 不标准化（默认，与 gplearn 一致）⭐推荐
    #   - 'simple': 简单除标准差（如果 gplearn 用了 norm 函数）
    #   - 'robust': log1p + z-score（会改变因子值）
    #   - 'zscore': 完整 z-score（会改变因子值）
    strategy.run_full_pipeline(
        weight_method='equal',
        normalize_method=None,  # 不标准化，与 gplearn 一致（推荐）
        enable_factor_selection=False  # 使用所有因子（推荐）
    )
    
    
    # ========== 方式3: 最简单的方式（无需YAML文件，推荐！） ==========
    # 定义因子表达式
    # factor_expressions = [
    #     'ta_dema_55(ta_dx_25(neg(h), ta_trix_8(h), ta_trix_21(ori_trix_21)))',
    #     # 'ta_lr_slope_20(ts_delta_17(ori_ta_macd))',
    #     # 'ta_rsi_14(close)',
    # ]
    
    # # 创建策略（只需指定核心参数）
    # strategy = QuantTradingStrategy.from_expressions_simple(
    #     factor_expressions=factor_expressions,
    #     sym='ETHUSDT',                                      # 交易对
    #     train_dates=('2025-01-01', '2025-01-20'),         # 训练集日期
    #     test_dates=('2025-01-20', '2025-01-31'),          # 测试集日期
    #     max_factors=5,                                     # 可选：最多使用5个因子
    #     fees_rate=0.0005,                                  # 可选：手续费率
    # )
    
    # # 运行
    # strategy.run_full_pipeline(weight_method='equal')
    
    
    # ========== 方式4: 链式调用（高级用法） ==========
    """
    # 如果需要更细粒度的控制，可以使用链式调用
    factor_expressions = ['ta_rsi_14(close)', 'ta_ema_20(close)']
    
    strategy = (QuantTradingStrategy
                .from_expressions_simple(factor_expressions, 
                                        sym='ETHUSDT',
                                        train_dates=('2025-01-01', '2025-02-01'),
                                        test_dates=('2025-02-01', '2025-03-01'))
                .load_data_from_dataload()
                .evaluate_factor_expressions()
                .normalize_factors()
                .select_factors()
                .prepare_training_data()
                .train_models()
                .make_predictions(weight_method='equal')
                .backtest_all_models())
    
    summary = strategy.get_performance_summary()
    """
    
    # ========== 可选：使用 Triple Barrier 标注 ==========
    # 在数据加载后、模型训练前，可以生成 Triple Barrier 标签
    # 
    # 示例：在因子筛选后添加 Triple Barrier
    # strategy = (QuantTradingStrategy.from_yaml(yaml_path, factor_csv_path, strategy_config)
    #            .load_data_from_dataload()
    #            .load_factor_expressions()
    #            .evaluate_factor_expressions()
    #            .normalize_factors()
    #            .select_factors()
    #            # 添加 Triple Barrier 标注
    #            .generate_triple_barrier_labels(
    #                pt_sl=[2, 2],         # 止盈/止损倍数
    #                max_holding=[0, 4]    # 最大持有4小时
    #            )
    #            # 可选：使用 Triple Barrier 的收益替代固定周期收益
    #            # .use_triple_barrier_as_y()
    #            .prepare_training_data()
    #            .train_models()
    #            .make_predictions(weight_method='equal')
    #            .backtest_all_models()
    # )
    
    # ========== 方式2: 手动配置（如果不用YAML） ==========
    # data_config = {
    #     'sym': 'ETHUSDT',
    #     'freq': '15m',
    #     'start_date_train': '2025-01-01',
    #     'end_date_train': '2025-01-20',
    #     'start_date_test': '2025-01-20',
    #     'end_date_test': '2025-01-31',
    #     'rolling_window': 2000,
    #     'data_dir': '/Volumes/Ext-Disk/data/futures/um/monthly/klines/ETHUSDT/15m',
    #     'read_frequency': 'monthly',
    #     'timeframe': '15m',
    #     'data_source': 'coarse_grain',  # 'kline', 'coarse_grain', 'micro', 'rolling'
    #     # coarse_grain 特有参数
    #     'coarse_grain_period': '2h',
    #     'feature_lookback_bars': 8,
    #     'rolling_step': '15min',
    #     'file_path': None,
    # }
    # 
    # config = {
    #     'return_period': 1,
    #     'corr_threshold': 0.5,
    #     'position_size': 1.0,
    #     'clip_num': 5.0,
    #     'fixed_return': 0.0,
    #     'fees_rate': 0.0005,
    #     'annual_bars': 365 * 24 * 4,
    #     'model_save_path': './models',
    #     'max_factors': 30,
    # }
    # 
    # factor_csv_path = 'ETHUSDT_15m_1_2025-01-01_2025-01-20_2025-01-20_2025-01-31.csv.gz'
    # 
    # strategy = QuantTradingStrategy(
    #     factor_csv_path=factor_csv_path,
    #     data_config=data_config,
    #     config=config
    # )
    # 
    # strategy.run_full_pipeline(weight_method='equal')
    
    # 绘制组合模型结果（推荐）
    strategy.plot_results('Ensemble')
    
    # 也可以查看单个模型
    # strategy.plot_results('LinearRegression')
    # strategy.plot_results('LightGBM')
    # strategy.plot_results('XGBoost')
    # strategy.plot_results('Ridge')
    # strategy.plot_results('Lasso')
    
    # 保存模型
    strategy.save_models()
    
    # 保持图形窗口显示
    plt.show(block=True)
