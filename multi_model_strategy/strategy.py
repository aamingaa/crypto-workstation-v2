"""
主策略类
整合所有模块，提供统一的接口
"""
import sys
from pathlib import Path
import itertools
import yaml
import numpy as np
import pandas as pd
import joblib
import pickle
import matplotlib.pyplot as plt
import os
from typing import Optional, List


# 确保项目路径和 gp_crypto_next 都在 sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 添加 gp_crypto_next 目录到 sys.path（解决其内部相对导入问题）
gp_crypto_dir = project_root / "gp_crypto_next"
if str(gp_crypto_dir) not in sys.path:
    sys.path.insert(0, str(gp_crypto_dir))

from multi_model_strategy.config import StrategyConfig, DataConfig
from multi_model_strategy.data_module import DataModule
from multi_model_strategy.factor_engine import FactorEngine
from multi_model_strategy.alpha_models import AlphaModelTrainer
from multi_model_strategy.position_scaling import PositionScalingManager, KellyBetSizer
from multi_model_strategy.backtest_engine import BacktestEngine
from multi_model_strategy.visualization import Visualizer
from multi_model_strategy.diagnostics import DiagnosticTools

# Triple Barrier（Lopez de Prado 风格）
from gp_crypto_next.triple_barrier import get_barrier, get_metalabel, cusum_filter


class QuantTradingStrategy:
    """
    多模型量化交易策略（整合 GP 因子）
    
    主要功能：
    1. 从表达式评估因子
    2. 多模型训练与集成
    3. Regime + Risk 层仓位缩放
    4. Kelly bet sizing（可选）
    5. Triple Barrier 标注（可选）
    6. 回测与诊断
    """
    
    # ========== 类方法：多种创建方式 ==========
    
    @classmethod
    def from_yaml_with_expressions(cls, yaml_path, factor_expressions, strategy_config=None):
        """从YAML配置 + 因子表达式创建"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        
        data_config = DataConfig.build_from_yaml(yaml_config)
        
        config = StrategyConfig.get_default_config()
        # 从 YAML 中同步部分策略相关参数
        config['return_period'] = yaml_config.get('y_train_ret_period', 1)
        # Triple Barrier / CUSUM 相关
        if 'triple_barrier_pt_sl' in yaml_config:
            config['triple_barrier_pt_sl'] = yaml_config['triple_barrier_pt_sl']
        if 'triple_barrier_max_holding' in yaml_config:
            config['triple_barrier_max_holding'] = yaml_config['triple_barrier_max_holding']
        if 'tb_cusum_h' in yaml_config:
            config['tb_cusum_h'] = yaml_config['tb_cusum_h']
        if 'tb_min_events' in yaml_config:
            config['tb_min_events'] = yaml_config['tb_min_events']
        if strategy_config:
            config.update(strategy_config)
        
        instance = cls(None, data_config, config)
        instance.factor_expressions = factor_expressions
        instance._use_expressions_mode = True
        
        print(f"已设置 {len(factor_expressions)} 个因子表达式")
        return instance
    
    @classmethod
    def from_expressions_simple(cls, factor_expressions, sym='ETHUSDT',
                                 train_dates=('2025-01-01', '2025-03-01'),
                                 test_dates=('2025-03-01', '2025-04-01'),
                                 **kwargs):
        """简化创建：只需因子表达式 + 基本参数"""
        data_config = DataConfig.build_simple(sym, kwargs.get('freq', '15m'),
                                              train_dates, test_dates, **kwargs)
        
        config = StrategyConfig.get_default_config()
        config['return_period'] = kwargs.get('return_period', 1)
        config['max_factors'] = kwargs.get('max_factors', 10)
        config['fees_rate'] = kwargs.get('fees_rate', 0.0005)
        
        instance = cls(None, data_config, config)
        instance.factor_expressions = factor_expressions
        instance._use_expressions_mode = True
        
        print(f"✓ 创建策略: {sym} | 训练: {train_dates[0]}~{train_dates[1]} | 测试: {test_dates[0]}~{test_dates[1]}")
        print(f"✓ 因子数量: {len(factor_expressions)}")
        
        return instance
    
    @classmethod
    def from_yaml(cls, yaml_path, factor_csv_path=None, strategy_config=None):
        """从YAML配置文件创建（支持从CSV加载因子）"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        
        data_config = DataConfig.build_from_yaml(yaml_config)
        
        config = StrategyConfig.get_default_config()
        config['return_period'] = yaml_config.get('y_train_ret_period', 1)
        if strategy_config:
            config.update(strategy_config)
        
        return cls(factor_csv_path, data_config, config)
    
    # ========== 初始化 ==========
    
    def __init__(self, factor_csv_path, data_config, config=None):
        """
        Args:
            factor_csv_path (str or None): GP因子CSV路径（可选）
            data_config (dict): 数据配置
            config (dict): 策略配置
        """
        self.factor_csv_path = factor_csv_path
        self.data_config = data_config
        self.config = config or StrategyConfig.get_default_config()
        
        # 数据相关
        self.factor_expressions = []
        self._use_expressions_mode = False
        
        # 子模块（延迟初始化）
        self.data_module = None
        self.factor_engine = None
        self.alpha_trainer = None
        self.position_manager = None
        self.backtest_engine = None
        self.visualizer = None
        self.diagnostic_tools = None
        
        # 数据容器
        self.X_all = None
        self.feature_names = None
        self.y_train = None
        self.y_test = None
        self.ret_train = None
        self.ret_test = None
        self.z_index = None
        
        # 结果容器
        self.predictions = {}
        self.backtest_results = {}
        self.ensemble_weights = None
        
        # Triple Barrier 相关
        self.barrier_results = None
        self.meta_labels = None
        
        print(f"策略初始化完成")
    
    # ========== 主流程 ==========
    

    def run_full_pipeline(self, is_pre_load=False, weight_method='equal', normalize_method=None,
                         enable_factor_selection=False, enable_quantile_features=False):
        """
        运行完整策略流程
        
        Args:
            weight_method (str): 'equal' 或 'sharpe'
            normalize_method (str or None): 因子标准化方法（None=不标准化）
            enable_factor_selection (bool): 是否启用因子筛选
            
            总结一下三者的“职责分工”
                Triple-barrier（bar 级，_apply_triple_barrier_labels）
                对象：所有 bar。
                用途：给 base 回归模型造连续 label 
                ​
                不关心：你实际是不是在这个 bar 下单。
                模型 train/predict + 离散化（_discretize_base_positions）
                对象：所有 bar 的预测。
                用途：把连续预测变成“真实下单/不下单”的仓位序列，定义交易规则。

                Trade-level TB（_build_trade_level_barriers_from_positions）
                对象：只有“0→非0”这些实际开仓事件。
                用途：评估每一笔真实交易的表现，生成 meta-label 和 Kelly 所需的 p,R
        """
        print("="*60)
        print("开始运行完整的量化策略流程（整合GP因子）")
        print("="*60)
        
        # 1. 数据加载
        if is_pre_load == False:
            self._load_data()
        
        # 2. 因子表达式加载
        if not self._use_expressions_mode and self.factor_csv_path:
            self._load_factor_expressions_from_csv()
        
        use_triple_barrier_label = self.config.get('use_triple_barrier_label', False)
        
        use_kelly_bet_sizing = self.config.get('use_kelly_bet_sizing', False)

        print(f"use_triple_barrier_label: {use_triple_barrier_label}, normalize_method: {normalize_method}, enable_factor_selection: {enable_factor_selection}, weight_method: {weight_method}, use_kelly_bet_sizing: {use_kelly_bet_sizing}")
        
        # 3. Triple Barrier（可选）
        if use_triple_barrier_label:
            self._apply_triple_barrier_labels()
        
        # 4. 因子评估
        self._evaluate_factors(normalize_method, enable_factor_selection)
        
        # 4.5（可选）：基于分箱权重为因子生成附加特征
        if enable_quantile_features:
            print("ℹ️ 启用因子分箱权重特征生成（auto_add_quantile_weighted_features）")
            self.auto_add_quantile_weighted_features(
                factor_names=None,
                data_range='train',
                n_quantiles=5,
                metric_key='Sharpe Ratio',
            )
        
        # 5. 模型训练与预测
        self._train_and_predict(weight_method)
        
        # 6. Kelly bet sizing（可选）
        if use_kelly_bet_sizing:
            self._apply_kelly_sizing()
        
        # 7. Regime & Risk 缩放
        self._apply_position_scaling()
        
        # 8. 回测
        self._run_backtest()
        
        print("\n" + "="*60)
        print("策略流程执行完成！")
        print("="*60)
        
        return self
    
    # ========== 内部步骤 ==========
    
    def _load_data(self):
        """步骤1：加载数据"""
        self.data_module = DataModule(self.data_config, self.config)
        self.data_module.load()
        
        # 提取数据
        data_dict = self.data_module.get_data_dict()
        self.X_all = data_dict['X_all']
        self.feature_names = data_dict['feature_names']
        self.y_train = data_dict['y_train']
        self.y_test = data_dict['y_test']
        self.ret_train = data_dict['ret_train']
        self.ret_test = data_dict['ret_test']
        self.z_index = data_dict['z_index']
        
        # 初始化可视化和诊断工具（需要价格数据）
        self.visualizer = Visualizer(
            str(self.data_module.get_total_factor_file_dir()),
            data_dict['z_index'],
            data_dict['close_train'],
            data_dict['close_test']
        )
        
        # 初始化回测引擎
        self.backtest_engine = BacktestEngine(
            data_dict['open_train'],
            data_dict['close_train'],
            data_dict['open_test'],
            data_dict['close_test'],
            self.config['fees_rate'],
            self.config['annual_bars']
        )
    
    def _load_factor_expressions_from_csv(self):
        """步骤2（可选）：从CSV加载因子表达式"""
        print(f"正在从 {self.factor_csv_path} 加载因子表达式...")
        
        factor_df = pd.read_csv(
            self.factor_csv_path,
            compression='gzip' if self.factor_csv_path.endswith('.gz') else None
        )
        
        if 'expression' not in factor_df.columns:
            raise ValueError("CSV文件中未找到'expression'列")
        
        all_expressions = factor_df['expression'].dropna().unique().tolist()
        
        # 根据性能筛选
        if 'fitness_sharpe_test' in factor_df.columns:
            print("根据测试集夏普比率筛选因子...")
            factor_df_sorted = factor_df.sort_values('fitness_sharpe_test', ascending=False)
            max_factors = min(self.config['max_factors'], len(factor_df_sorted))
            self.factor_expressions = factor_df_sorted['expression'].head(max_factors).tolist()
        else:
            max_factors = min(self.config['max_factors'], len(all_expressions))
            self.factor_expressions = all_expressions[:max_factors]
        
        print(f"筛选出 {len(self.factor_expressions)} 个因子")
    
    def _apply_triple_barrier_labels(self):
        """步骤3（可选）：应用 Triple Barrier 标注"""
        print("ℹ️ 使用 Triple Barrier 收益作为回归标签")
        
        pt_sl = self.config.get('triple_barrier_pt_sl', [2, 2])
        max_holding = self.config.get('triple_barrier_max_holding', [0, 4])
        
        # 生成 TB 标签所需的价格与目标波动率
        close_series, target_volatility = self._prepare_triple_barrier_price_and_vol()
        
        # ========== 事件选择：Lopez 原书 CUSUM 版本 ==========
        # 1) 计算 log price 的增量，用于 CUSUM
        log_price = np.log(close_series.astype(float))
        diff = log_price.diff().dropna()
        if len(diff) == 0:
            # 极端兜底：价格序列过短或常数，退回每个 bar 作为事件
            enter_points = close_series.index
            print("⚠️ CUSUM 无有效 diff，退回使用所有 bar 作为事件")
        else:
            # 2) 阈值 h：默认为 h * std(diff)，h 可通过 config 调整
            h = float(self.config.get('tb_cusum_h', 3.0))
            threshold = h * diff.std()
            if threshold <= 0:
                threshold = diff.abs().median()
            
            enter_points = cusum_filter(log_price, threshold)
            min_events = int(self.config.get('tb_min_events', 500))
            if len(enter_points) < min_events:
                print(f"⚠️ CUSUM 事件数过少({len(enter_points)}<{min_events})，退回使用所有 bar 作为事件")
                enter_points = close_series.index
        
        print(f"Triple Barrier 使用事件数: {len(enter_points)} / 总bar数: {len(close_series)}")
        
        # 3) 调用 Triple Barrier：仅用于构造回归标签（y_train / y_test），不用于 Kelly / meta-label
        self.barrier_results = get_barrier(
            close=close_series,
            enter=enter_points,
            pt_sl=pt_sl,
            max_holding=max_holding,
            target=target_volatility,
            side=pd.Series(1.0, index=enter_points)
        )
        
        # 替换 label
        barrier_ret = self.barrier_results['ret']
        train_idx = getattr(self.data_module, 'train_index', None)
        test_idx = getattr(self.data_module, 'test_index', None)

        if train_idx is not None and test_idx is not None:
            # 使用时间索引对齐（coarse_grain 模式）
            y_train_tb = barrier_ret.reindex(train_idx)
            y_test_tb = barrier_ret.reindex(test_idx)

            if y_train_tb.isna().any() or y_test_tb.isna().any():
                raise ValueError("Triple Barrier 标签在 train_index/test_index 上存在 NaN，请检查时间对齐。")

            self.y_train = y_train_tb.values.reshape(-1, 1)
            self.y_test = y_test_tb.values.reshape(-1, 1)
        else:
            # 兼容 kline 等旧模式：按长度切分
            barrier_ret_values = barrier_ret.values
            n_train = len(self.y_train)
            n_test = len(self.y_test)
            if len(barrier_ret_values) < n_train + n_test:
                raise ValueError(
                    f"Triple Barrier 返回长度 {len(barrier_ret_values)} < 训练+测试长度 {n_train + n_test}"
                )
            self.y_train = barrier_ret_values[:n_train].reshape(-1, 1)
            self.y_test = barrier_ret_values[n_train:n_train + n_test].reshape(-1, 1)
        
        self.ret_train = self.y_train.ravel()
        self.ret_test = self.y_test.ravel()
        
        print(f"Triple Barrier 标签应用完成（基于所有 bar 的标签，用于回归模型）")
    
    def _prepare_triple_barrier_price_and_vol(self):
        """
        提取用于 Triple Barrier 的价格序列与目标波动率。
        
        Returns:
            tuple(pd.Series, pd.Series): (close_series, target_volatility)
        """
        # 优先使用 DataModule 中对齐好的 ohlc DataFrame（包含完整样本期的价格）
        ohlc = self.data_module.ohlc
        if 'close' in ohlc.columns:
            close_series = ohlc['close'].astype(float)
        elif 'c' in ohlc.columns:
            close_series = ohlc['c'].astype(float)
        else:
            # 回退：使用第一列作为收盘价
            close_series = ohlc.iloc[:, 0].astype(float)
        
        # 确保索引为日期时间类型
        close_series.index = pd.to_datetime(close_series.index)
        
        rolling_window = self.data_config.get('rolling_window', 2000)
        target_volatility = close_series.pct_change().rolling(
            window=min(rolling_window, len(close_series) // 2)
        ).std()
        target_volatility = target_volatility.fillna(method='bfill')
        
        return close_series, target_volatility
    
    def _build_trade_level_barriers_from_positions(self):
        """
        基于「离散化后的实际仓位开仓时刻」构建 trade-level Triple Barrier 与 meta-label，
        主要用于 Kelly / meta-labeling，而不再覆盖回归模型的 y_train / y_test。
        """
        if not self.predictions:
            print("当前还没有模型预测结果，无法基于仓位构建 Triple Barrier")
            return
        
        pt_sl = self.config.get('triple_barrier_pt_sl', [2, 2])
        max_holding = self.config.get('triple_barrier_max_holding', [0, 4])
        
        # 1) 价格与波动率（与 _apply_triple_barrier_labels 保持一致）
        close_series, target_volatility = self._prepare_triple_barrier_price_and_vol()
        
        # 2) 构造完整时间索引（train + test），与模型预测对齐
        train_len = len(self.y_train)
        test_len = len(self.y_test)
        
        if getattr(self.data_module, 'train_index', None) is not None and \
           getattr(self.data_module, 'test_index', None) is not None:
            idx_train = pd.to_datetime(self.data_module.train_index)
            idx_test = pd.to_datetime(self.data_module.test_index)
            full_index = idx_train.append(idx_test)
        else:
            full_index = pd.to_datetime(self.z_index[:train_len + test_len])
        
        # 3) 选择基准模型的离散仓位（优先使用 Ensemble）
        if 'Ensemble' in self.predictions:
            base_name = 'Ensemble'
        else:
            base_name = next(iter(self.predictions.keys()))
        
        base_train = np.asarray(self.predictions[base_name]['train']).flatten()
        base_test = np.asarray(self.predictions[base_name]['test']).flatten()
        pos_all = np.concatenate([base_train, base_test])
        
        if len(pos_all) != len(full_index):
            min_len = min(len(pos_all), len(full_index))
            pos_all = pos_all[:min_len]
            full_index = full_index[:min_len]
        
        pos_series = pd.Series(pos_all, index=full_index)
        
        # 将仓位与价格对齐在共同索引上
        common_index = close_series.index.intersection(pos_series.index)
        if len(common_index) == 0:
            print("⚠️ 仓位时间索引与价格索引没有交集，无法构建 trade-level Triple Barrier")
            return
        
        pos_series = pos_series.reindex(common_index).fillna(0.0)
        close_aligned = close_series.reindex(common_index)
        target_aligned = target_volatility.reindex(common_index).fillna(method='bfill')
        
        # 4) 找到「从 0 → 非 0」的开仓时刻作为 enter，方向由仓位符号决定
        prev_pos = pos_series.shift(1).fillna(0.0)
        enter_mask = (prev_pos == 0.0) & (pos_series != 0.0)
        enter_times = pos_series.index[enter_mask]
        
        if len(enter_times) == 0:
            print("⚠️ 离散仓位中没有任何 0→非0 的开仓事件，无法构建 trade-level Triple Barrier")
            return
        
        side_series = np.sign(pos_series.loc[enter_times])
        
        # 5) 基于真实开仓事件构建 Triple Barrier
        barrier_trades = get_barrier(
            close=close_aligned,
            enter=enter_times,
            pt_sl=pt_sl,
            max_holding=max_holding,
            target=target_aligned,
            side=side_series
        )
        self.barrier_results = barrier_trades
        
        # 6) 构建 meta-label，并对齐到完整时间轴（train+test），
        #    非开仓 bar 上的 meta-label 设为 NaN，Kelly 训练时仅使用有标签的样本。
        meta_series = get_metalabel(barrier_trades)  # index 为事件起点（enter_times 子集，且 ret!=0）
        meta_full = pd.Series(np.nan, index=common_index)
        meta_full.loc[meta_series.index] = meta_series.values
        meta_full = meta_full.reindex(full_index)
        self.meta_labels = meta_full.values
        
        print(f"Trade-level Triple Barrier 构建完成：开仓笔数={len(barrier_trades)}")
    
    def _discretize_base_positions(self):
        """
        基于预测强度对 base 模型信号做简单阈值离散化：
        只在“强信号” bar 上持仓，其余视为不交易。
        """
        if not self.predictions:
            return
        
        strength_pct = self.config.get('signal_strength_pct', None)
        if strength_pct is None:
            print("不做信号阈值离散化（signal_strength_pct=None）")
            return
        
        try:
            strength_pct = float(strength_pct)
        except (TypeError, ValueError):
            print(f"signal_strength_pct 配置非法: {self.config.get('signal_strength_pct')}, 跳过离散化")
            return
        
        if not (0.0 < strength_pct < 100.0):
            print(f"signal_strength_pct={strength_pct} 不在 (0, 100) 内，跳过离散化")
            return
        
        # 选择基准模型（优先使用 Ensemble）
        if 'Ensemble' in self.predictions:
            base_name = 'Ensemble'
        else:
            base_name = next(iter(self.predictions.keys()))
        
        base_train = np.asarray(self.predictions[base_name]['train']).flatten()
        abs_train = np.abs(base_train)
        abs_train = abs_train[np.isfinite(abs_train)]
        
        if abs_train.size == 0:
            print("训练预测为空或全部为 NaN，无法计算信号阈值，跳过离散化")
            return
        
        thr = np.percentile(abs_train, strength_pct)
        print(f"应用信号阈值离散化：使用 {base_name} 训练预测绝对值 "
              f"{strength_pct:.1f}% 分位数作为阈值 {thr:.6f}")
        
        def _to_pos(arr):
            arr = np.asarray(arr).flatten()
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            sign = np.sign(arr)
            strong = (np.abs(arr) >= thr).astype(float)
            return sign * strong
        
        # 对所有模型的 train/test 仓位统一应用离散化
        for model_name, pred in self.predictions.items():
            self.predictions[model_name]['train'] = _to_pos(pred['train'])
            self.predictions[model_name]['test'] = _to_pos(pred['test'])
    
    def _evaluate_factors(self, normalize_method, enable_factor_selection):
        """步骤4：评估因子"""
        self.factor_engine = FactorEngine(
            self.factor_expressions,
            self.X_all,
            self.feature_names,
            self.y_train,
            rolling_window=self.data_config.get('rolling_window', 200),
            index=self.z_index  # 使用时间索引对齐因子与样本
        )
        
        self.factor_engine.evaluate_expressions()
        
        # 可选：标准化
        if normalize_method:
            print(f"⚠️  警告: 启用因子标准化可能导致效果与 gplearn 不一致！")
            self.factor_engine.normalize(method=normalize_method)
        else:
            print("ℹ️  不标准化因子（与 gplearn 一致）")
        
        # 可选：因子筛选
        if enable_factor_selection:
            self.factor_engine.select_by_correlation(self.config['corr_threshold'])
    
    def _train_and_predict(self, weight_method):
        """步骤5：训练模型与预测"""
        selected_factors = self.factor_engine.get_selected_factors()
        factor_data = self.factor_engine.get_factor_data()
        
        # 使用 dataloader 返回的时间索引来对齐训练集 / 测试集，避免中间 gap 带来的错位
        train_index = getattr(self.data_module, 'train_index', None)
        test_index = getattr(self.data_module, 'test_index', None)
        
        if train_index is not None and test_index is not None:
            
            X_train = factor_data.loc[train_index, selected_factors].values
            X_test = factor_data.loc[test_index, selected_factors].values
            
            print(f"使用时间索引切分因子数据: X_train={X_train.shape}, X_test={X_test.shape}")
        else:
            # 回退到旧逻辑（仅用于 kline 等不返回索引的模式）
            train_len = len(self.y_train)
            test_len = len(self.y_test)
            
            print(f"[fallback] X_all len: {len(self.X_all)}, train_len: {train_len}, test_len: {test_len}")
            X_train = factor_data[selected_factors].iloc[:train_len].values
            X_test = factor_data[selected_factors].iloc[-test_len:].values
        
        # 初始化诊断工具（需要因子数据）
        self.diagnostic_tools = DiagnosticTools(
            factor_data, selected_factors,
            self.ret_train, self.ret_test,
            self.y_train, self.y_test,
            self.data_module.open_train, self.data_module.close_train,
            self.data_module.open_test, self.data_module.close_test,
            self.config['fees_rate'], self.config['annual_bars']
        )
        
        self.alpha_trainer = AlphaModelTrainer(
            X_train, X_test, self.y_train, self.y_test, selected_factors
        )

        # self.alpha_trainer.apply_pca(explained_var_ratio=0.95, standardize=True)

        
        self.alpha_trainer.train_all_models(use_normalized_label=True)
        self.alpha_trainer.make_predictions()
        
        self.alpha_trainer.plot_training_losses(
            output_dir=self.data_module.get_total_factor_file_dir()  # 或任何你想存的目录
        )
        # 模型集成
        backtest_fn = lambda pos, data_range: self.backtest_engine.run_backtest(pos, data_range)
        self.alpha_trainer.ensemble_models(weight_method, backtest_fn)
        
        self.predictions = self.alpha_trainer.get_predictions()
        self.ensemble_weights = self.alpha_trainer.ensemble_weights
        
        # 基于预测强度做一次统一的信号阈值离散化（改进 base enter 逻辑）
        self._discretize_base_positions()
        
    def _apply_kelly_sizing(self):
        """步骤6（可选）：Kelly bet sizing"""
        print("ℹ️ 启用 Lopez 风格 Kelly bet sizing")
        
        # 基于「离散化后的 base 仓位」构建 trade-level Triple Barrier 与 meta-label
        # 用于估计胜率 p 与盈亏比 R
        self._build_trade_level_barriers_from_positions()
        
        kelly_sizer = KellyBetSizer(self.config)
        
        train_len = len(self.y_train)
        X_train = list(self.predictions.values())[0]['train']  # 用第一个模型的特征维度
        X_test = list(self.predictions.values())[0]['test']
        
        kelly_sizer.train_meta_model(
            self.alpha_trainer.X_train,
            self.alpha_trainer.X_test,
            self.meta_labels,
            train_len
        )
        kelly_sizer.compute_R_from_barrier(self.barrier_results, train_len)
        self.predictions = kelly_sizer.apply_kelly_sizing(self.predictions, 'Ensemble')
    
    def _apply_position_scaling(self):
        """步骤7：Regime & Risk 缩放"""
        feature_df = pd.DataFrame(self.X_all, columns=self.feature_names)
        train_len = len(self.y_train)
        
        self.position_manager = PositionScalingManager(
            self.config, feature_df, train_len
        )
        
        self.position_manager.build_regime_and_risk_scalers()
        self.predictions = self.position_manager.apply_to_predictions(self.predictions)
    
    def _run_backtest(self):
        """步骤8：回测"""
        self.backtest_results = self.backtest_engine.backtest_all_models(self.predictions)
        self.backtest_engine.get_performance_summary()
    
    # ========== 小区间诊断接口 ==========
    
    def backtest_subperiod(self, start_bar, end_bar, model_name='Ensemble', data_range='test'):
        """
        小区间回测：对指定 bar 区间的仓位做单独回测与诊断
        
        Args:
            start_bar (int): 相对于 data_range 起点的起始 bar（含）
            end_bar (int): 相对于 data_range 起点的结束 bar（不含）
            model_name (str): 模型名称
            data_range (str): 'train' 或 'test'
        
        Returns:
            tuple: (pnl_sub, metrics_sub)
        """
        if not self.predictions:
            print("请先完成模型训练与整体回测")
            return None, None
        
        if model_name not in self.predictions:
            print(f"模型 {model_name} 的预测结果不存在")
            return None, None
        
        if data_range not in ('train', 'test'):
            raise ValueError("data_range 必须是 'train' 或 'test'")
        
        pos = self.predictions[model_name][data_range]
        pnl_sub, metrics_sub = self.backtest_engine.run_backtest_subperiod(
            pos, data_range=data_range, start_bar=start_bar, end_bar=end_bar
        )
        
        print(f"\n===== 小区间回测结果 | 模型: {model_name} | 区间: {data_range}[{start_bar}:{end_bar}] =====")
        for k, v in metrics_sub.items():
            if "Rate" in k or "Ratio" in k or "Return" in k or "Drawdown" in k:
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print("===== 小区间回测结束 =====\n")
        
        return pnl_sub, metrics_sub
    
    def backtest_subperiod_by_time(self, start_time, end_time, model_name='Ensemble', data_range='test'):
        """
        按时间区间做小区间回测（内部自动映射为 bar 区间）
        
        Args:
            start_time (str or pd.Timestamp): 区间起始时间（含），如 '2025-02-01 00:00:00'
            end_time (str or pd.Timestamp): 区间结束时间（不含）
            model_name (str): 模型名称
            data_range (str): 'train' 或 'test'
        
        Returns:
            tuple: (pnl_sub, metrics_sub)
        """
        if self.data_module is None or self.z_index is None:
            print("数据模块尚未初始化，无法按时间做小区间回测")
            return None, None
        
        if data_range not in ('train', 'test'):
            raise ValueError("data_range 必须是 'train' 或 'test'")
        
        # 时间标准化
        start_ts = pd.to_datetime(start_time)
        end_ts = pd.to_datetime(end_time)
        
        # 取对应区间的时间索引
        train_len = len(self.y_train)
        test_len = len(self.y_test)
        
        if getattr(self.data_module, 'train_index', None) is not None and \
           getattr(self.data_module, 'test_index', None) is not None:
            # coarse_grain 路径：直接使用 train_index / test_index
            if data_range == 'train':
                idx_range = self.data_module.train_index
            else:
                idx_range = self.data_module.test_index
        else:
            # kline 等路径：用 z_index 按长度切分
            if data_range == 'train':
                idx_range = pd.to_datetime(self.z_index[:train_len])
            else:
                idx_range = pd.to_datetime(self.z_index[train_len:train_len + test_len])
        
        # 根据时间找到对应的 bar 区间
        idx_range = pd.to_datetime(idx_range)
        mask = (idx_range >= start_ts) & (idx_range < end_ts)
        pos_indices = np.where(mask)[0]
        
        if len(pos_indices) == 0:
            print(f"在 {data_range} 段内未找到时间区间 [{start_ts}, {end_ts}) 对应的样本")
            return None, None
        
        start_bar = int(pos_indices[0])
        end_bar = int(pos_indices[-1] + 1)
        
        print(f"时间区间 [{start_ts}, {end_ts}) 映射到 {data_range} 段 bar 区间 [{start_bar}:{end_bar}]")
        pnl_sub, metrics_sub = self.backtest_subperiod(
            start_bar, end_bar, model_name=model_name, data_range=data_range
        )

        # 构造与 pnl_sub 对齐的时间索引与价格
        idx_sub = idx_range[start_bar:end_bar]
        if data_range == 'train':
            price_all = self.data_module.close_train
        else:
            price_all = self.data_module.close_test

        # 价格可能是 Series 或 ndarray
        if isinstance(price_all, pd.Series):
            price_vals = price_all.values
        else:
            price_vals = np.asarray(price_all)
        price_sub = price_vals[start_bar:end_bar]

        # 绘制价格 + 累计PnL
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(idx_sub, price_sub, 'b-', linewidth=1.2, label='价格')
        ax1.set_xlabel('时间')
        ax1.set_ylabel('价格', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3, linestyle='--')

        ax2 = ax1.twinx()
        ax2.plot(idx_sub, pnl_sub, 'r-', linewidth=1.5, label='累计PnL')
        ax2.set_ylabel('累计收益', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        plt.title(f'子区间价格与累计PnL [{start_time} ~ {end_time}]')

        # 合并图例
        lines = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        fig.tight_layout()

        file_path = f"{self.data_module.get_total_factor_file_dir()}/model_drawings/subperiod/{start_ts}_{end_ts}/backtest_results.png"
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path)
        
        plt.close()
        # plt.show()

        return pnl_sub, metrics_sub
    
    # ========== 可视化接口 ==========
    
    def plot_results(self, model_name='Ensemble'):
        """绘制回测结果"""
        self.visualizer.plot_backtest_results(
            self.backtest_results, model_name, self.ensemble_weights
        )
        return self
    
    def plot_regime_and_risk(self, model_name='Ensemble'):
        """绘制 Regime & Risk 诊断图"""
        if self.position_manager:
            self.visualizer.plot_regime_and_risk_scalers(
                self.predictions,
                self.position_manager.regime_scaler_train,
                self.position_manager.regime_scaler_test,
                self.position_manager.risk_scaler_train,
                self.position_manager.risk_scaler_test,
                model_name
            )
        return self
    
    # ========== 诊断接口 ==========
    
    def diagnose_label_health(self):
        """Label 健康度诊断"""
        if self.diagnostic_tools:
            self.diagnostic_tools.diagnose_label_health(
                self.barrier_results, self.meta_labels
            )
        return self
    
    def diagnose_tb_events(self, max_points: int = 2000):
        """
        简单诊断 Triple Barrier 事件分布：
        在价格曲线上叠加 CUSUM 事件点，直观查看事件是否过稀/过密。
        
        Args:
            max_points (int): 若价格点过多，最多采样展示的点数（避免图太密）
        """
        if self.data_module is None:
            print("数据模块尚未初始化，无法诊断 TB 事件")
            return self
        
        close_series, _ = self._prepare_triple_barrier_price_and_vol()
        if len(close_series) == 0:
            print("收盘价序列为空，无法诊断 TB 事件")
            return self
        
        # 使用与 _apply_triple_barrier_labels 相同的 CUSUM 规则
        log_price = np.log(close_series.astype(float))
        diff = log_price.diff().dropna()
        if len(diff) == 0:
            print("⚠️ CUSUM 无有效 diff，事件退回为所有 bar")
            enter_points = close_series.index
        else:
            h = float(self.config.get('tb_cusum_h', 3.0))
            threshold = h * diff.std()
            if threshold <= 0:
                threshold = diff.abs().median()
            enter_points = cusum_filter(log_price, threshold)
        
        print(f"[TB 诊断] 总 bar 数: {len(close_series)}, CUSUM 事件数: {len(enter_points)}")
        
        # 采样避免图太密
        if len(close_series) > max_points:
            step = len(close_series) // max_points
            close_plot = close_series.iloc[::step]
        else:
            close_plot = close_series
        
        plt.figure(figsize=(12, 6))
        plt.plot(close_plot.index, close_plot.values, label='Close', alpha=0.7)
        
        # 仅在采样范围内绘制事件点
        enter_in_range = [t for t in enter_points if t in close_plot.index]
        if len(enter_in_range) > 0:
            plt.scatter(enter_in_range,
                        close_plot.loc[enter_in_range],
                        color='red', s=10, label='CUSUM Events')
        
        plt.title("Triple Barrier CUSUM Events vs Price")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return self
    
    def diagnose_factor_ic(self, data_range='train', top_n=20):
        """因子 IC 诊断"""
        if self.diagnostic_tools:
            return self.diagnostic_tools.diagnose_factor_ic(data_range, top_n)
        return None
    
    def backtest_single_factor(self, factor_name, data_range='test', n_quantiles=5):
        """单因子回测"""
        if self.diagnostic_tools:
            return self.diagnostic_tools.backtest_single_factor(
                factor_name, data_range, n_quantiles
            )
        return None
    
    def diagnose_top_factors(self, data_range='test', top_n=5):
        """Top 因子回测"""
        if self.diagnostic_tools:
            return self.diagnostic_tools.diagnose_top_factors_backtest(
                data_range, top_n
            )
        return None
    
    def optimize_factor_quantile_weights(
        self,
        factor_name: str,
        data_range: str = 'train',
        n_quantiles: int = 5,
        schemes: dict = None,
        metric_key: str = 'Sharpe Ratio',
    ) -> pd.DataFrame:
        """
        基于少量预设的分箱权重方案，对单因子做「按分箱权重」策略回测，用于自动选出较优方案。
        
        Args:
            factor_name (str): 因子名称（必须在 factor_data 列中）
            data_range (str): 'train' 或 'test'，用于评估权重方案
            n_quantiles (int): 分箱数量（与 DiagnosticTools 中保持一致）
            schemes (dict or None): {scheme_name: {quantile_index: weight}}；
                                   若为 None，则使用若干常见模板
            metric_key (str): 评价指标（如 'Sharpe Ratio', 'Annual Return', 'Calmar Ratio'）
        
        Returns:
            pd.DataFrame: 每个方案对应的回测指标，按 metric_key 降序排序
        """
        # 确保 diagnostic_tools 已初始化；若尚未初始化，则基于当前 factor_engine/data_module 构建
        if self.diagnostic_tools is None:
            if self.factor_engine is None or self.data_module is None:
                print("diagnostic_tools 尚未初始化，且因子/数据模块为空，请先运行 _load_data 和 _evaluate_factors。")
                return None
            factor_data = self.factor_engine.get_factor_data()
            selected_factors = self.factor_engine.get_selected_factors()
            self.diagnostic_tools = DiagnosticTools(
                factor_data,
                selected_factors,
                self.ret_train,
                self.ret_test,
                self.y_train,
                self.y_test,
                self.data_module.open_train,
                self.data_module.close_train,
                self.data_module.open_test,
                self.data_module.close_test,
                self.config['fees_rate'],
                self.config['annual_bars'],
            )
        
        if schemes is None:
            # 一些常见、形状合理的模板（可按需扩展）
            schemes = {
                "long_short_extreme": {0: -1.0, n_quantiles - 1: 1.0},      # 多最高、空最低
                "long_only_extreme": {n_quantiles - 1: 1.0},                # 只多最高
                "long_middle": {n_quantiles // 2: 1.0},                     # 只多中间
                "long_middle_high": {n_quantiles // 2: 0.5, n_quantiles - 1: 1.0},  # 中高权重
            }
        
        records = []
        for scheme_name, weights in schemes.items():
            res = self.diagnostic_tools.backtest_single_factor_by_quantile_weights(
                factor_name=factor_name,
                weights=weights,
                data_range=data_range,
                n_quantiles=n_quantiles,
            )
            if res is None:
                continue
            _, metrics = res
            rec = {
                "factor": factor_name,
                "scheme": scheme_name,
            }
            rec.update(metrics)
            records.append(rec)
        
        if not records:
            print(f"未能对因子 {factor_name} 评估任何分箱权重方案。")
            return None
        
        df = pd.DataFrame(records)
        if metric_key in df.columns:
            df = df.sort_values(metric_key, ascending=False)
        else:
            print(f"⚠️ 指标 {metric_key} 不在结果列中，实际列: {list(df.columns)}，返回未排序结果。")
        
        # 自动保存一份 csv 便于后续查看
        try:
            out_dir = Path(self.data_module.get_total_factor_file_dir()) / "diagnostics"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"quantile_weight_schemes_{factor_name}.csv"
            # 简单清理文件名
            out_path = Path(str(out_path).replace("/", "_").replace("\\", "_").replace(" ", "_"))
            df.to_csv(out_path, index=False)
            print(f"分箱权重方案评估结果已保存至: {out_path}")
        except Exception as e:
            print(f"保存分箱权重方案评估结果失败: {e}")
        
        return df

    def add_quantile_weighted_feature(
        self,
        factor_name: str,
        weights: dict,
        n_quantiles: int = 5,
        new_name: str  = None,
    ):
        """
        使用分箱权重，将原始因子加工成一条新的 signal 特征，并加入因子数据/选中因子列表。
        """
        if self.factor_engine is None:
            print("factor_engine 尚未初始化，请先运行 _evaluate_factors。")
            return
        if self.diagnostic_tools is None:
            print("diagnostic_tools 尚未初始化，请先调用 optimize_factor_quantile_weights 或构建 DiagnosticTools。")
            return

        factor_data = self.factor_engine.get_factor_data()
        sig = self.diagnostic_tools.build_quantile_weighted_signal(
            factor_name=factor_name,
            weights=weights,
            n_quantiles=n_quantiles,
        )

        col_name = new_name or f"{factor_name}_qw"
        factor_data[col_name] = sig.values
        self.factor_engine.factor_data = factor_data

        selected = self.factor_engine.get_selected_factors()
        if col_name not in selected:
            selected.append(col_name)
            self.factor_engine.selected_factors = selected

        print(f"已添加分箱权重特征列: {col_name}")

    def auto_add_quantile_weighted_features(
        self,
        factor_names: Optional[List[str]] = None,
        data_range: str = 'train',
        n_quantiles: int = 5,
        metric_key: str = 'Sharpe Ratio',
    ):
        """
        对一批因子自动搜索若干分箱权重方案，在指定区间上按 metric_key 选出最佳方案，
        并将对应的分箱权重 signal 作为新特征加入因子矩阵。
        """
        if self.factor_engine is None:
            print("factor_engine 尚未初始化，请先运行 _evaluate_factors。")
            return

        # 默认对当前选中的所有因子做处理
        if factor_names is None:
            factor_names = self.factor_engine.get_selected_factors()

        # 使用与 optimize_factor_quantile_weights 相同的默认 schemes
        base_schemes = {
            "long_short_extreme": {0: -1.0, n_quantiles - 1: 1.0},
            "long_only_extreme": {n_quantiles - 1: 1.0},
            "long_middle": {n_quantiles // 2: 1.0},
            "long_middle_high": {n_quantiles // 2: 0.5, n_quantiles - 1: 1.0},
        }

        for f in factor_names:
            df = self.optimize_factor_quantile_weights(
                factor_name=f,
                data_range=data_range,
                n_quantiles=n_quantiles,
                schemes=base_schemes,
                metric_key=metric_key,
            )
            if df is None or df.empty:
                continue
            best_scheme_name = df.iloc[0]["scheme"]
            weights = base_schemes.get(best_scheme_name)
            if weights is None:
                continue
            new_name = f"{f}_qw_{best_scheme_name}"
            self.add_quantile_weighted_feature(
                factor_name=f,
                weights=weights,
                n_quantiles=n_quantiles,
                new_name=new_name,
            )
    
    # ========== 保存接口 ==========
    
    def save_models(self, save_path=None):
        """保存模型与诊断信息"""
        save_path = Path(save_path or self.config['model_save_path'])
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"正在保存模型到 {save_path}...")
        
        # 保存模型
        models = self.alpha_trainer.get_models()
        for model_name, model in models.items():
            if model_name == 'LightGBM':
                model.save_model(str(save_path / 'lgb_model.txt'))
            else:
                joblib.dump(model, str(save_path / f'{model_name.lower()}_model.pkl'))
        
        # 保存诊断信息
        diagnostics = {
            'regime_scaler_train': self.position_manager.regime_scaler_train if self.position_manager else None,
            'regime_scaler_test': self.position_manager.regime_scaler_test if self.position_manager else None,
            'risk_scaler_train': self.position_manager.risk_scaler_train if self.position_manager else None,
            'risk_scaler_test': self.position_manager.risk_scaler_test if self.position_manager else None,
            'predictions': {k: {'train': np.asarray(v['train']), 'test': np.asarray(v['test'])}
                           for k, v in self.predictions.items()},
            'pnl': {k: {'train_pnl': np.asarray(v['train_pnl']), 'test_pnl': np.asarray(v['test_pnl'])}
                   for k, v in self.backtest_results.items()},
            'z_index': np.asarray(self.z_index) if self.z_index is not None else None,
        }
        
        with open(save_path / 'diagnostics.pkl', 'wb') as f:
            pickle.dump(diagnostics, f)
        
        print(f"保存完成: {save_path}")
        return self
    
    def get_performance_summary(self):
        """获取绩效汇总"""
        return self.backtest_engine.get_performance_summary()

    # ========== 参数搜索（网格搜索）接口 ==========
    
    def run_param_search(
        self,
        signal_grid=None,
        pt_sl_grid=None,
        max_holding_grid=None,
        metric=None,
        data_range=None,
        model_name=None,
        weight_method='equal',
        normalize_method=None,
        enable_factor_selection=False,
        save_plots=False,
        plot_subdir='param_search',
    ):
        """
        简单网格搜索：在给定参数网格上跑完整策略流程，并基于样本外指标选出最优组合。
        
        Args:
            signal_grid (list[float] or None): signal_strength_pct 网格（分位数），None 则读取配置
            pt_sl_grid (list[list[float]] or None): Triple Barrier pt_sl 网格，例如 [[1,2],[2,2]]
            max_holding_grid (list[list[int]] or None): Triple Barrier max_holding 网格，例如 [[0,4],[0,8]]
            metric (str or None): 评价指标键，默认读取配置，可选 'Sharpe Ratio'、'Calmar Ratio'、'Annual Return'
            data_range (str or None): 'train' 或 'test'，用于评估超参数
            model_name (str or None): 使用哪个模型的回测指标，默认读取配置或 'Ensemble'
            weight_method, normalize_method, enable_factor_selection:
                传给 run_full_pipeline 的其它参数
            save_plots (bool): 若为 True，则为每个网格组合单独保存一张回测图
            plot_subdir (str): 保存网格搜索图片的子目录名（位于 total_factor_file_dir 下）
        
        Returns:
            dict: 包含最佳参数与对应指标的结果字典
        """
        # 1) 读取默认网格与参数
        cfg = self.config
        signal_grid = signal_grid or cfg.get('param_search_signal_strength_grid', [cfg.get('signal_strength_pct', 70.0)])
        pt_sl_grid = pt_sl_grid or cfg.get('param_search_triple_barrier_pt_sl_grid', [cfg.get('triple_barrier_pt_sl', [2.0, 2.0])])
        max_holding_grid = max_holding_grid or cfg.get('param_search_triple_barrier_max_holding_grid', [cfg.get('triple_barrier_max_holding', [0, 12])])
        
        metric = metric or cfg.get('param_search_metric', 'Sharpe Ratio')
        data_range = data_range or cfg.get('param_search_data_range', 'test')
        model_name = model_name or cfg.get('param_search_model_name', 'Ensemble')
        
        metric_key = str(metric)
        data_key = 'train' if data_range == 'train' else 'test'
        
        print("=" * 60)
        print(f"开始参数网格搜索：metric={metric_key}, data_range={data_key}, model={model_name}")
        print(f"signal_grid={signal_grid}")
        print(f"pt_sl_grid={pt_sl_grid}")
        print(f"max_holding_grid={max_holding_grid}")
        print("=" * 60)
        
        # 2) 备份原始配置
        orig_config = dict(self.config)
        
        best_score = None
        best_params = None
        best_summary = None
        
        self._load_data()
        is_pre_load = True
    
        # 3) 遍历所有组合
        for sig, pt_sl, max_h in itertools.product(signal_grid, pt_sl_grid, max_holding_grid):
            # 兼容 pt_sl / max_h 为 tuple 的情况
            pt_sl_list = list(pt_sl)
            max_h_list = list(max_h)
            
            self.config['signal_strength_pct'] = float(sig)
            self.config['triple_barrier_pt_sl'] = pt_sl_list
            self.config['triple_barrier_max_holding'] = max_h_list
            
            print(f"\n>>> 尝试参数组合: signal_strength_pct={sig}, "
                  f"pt_sl={pt_sl_list}, max_holding={max_h_list}")
            
            # 跑完整流程（包括 TB / 模型 / Kelly / Regime / 回测）
            self.run_full_pipeline(
                is_pre_load=is_pre_load,
                weight_method=weight_method,
                normalize_method=normalize_method,
                enable_factor_selection=enable_factor_selection,
            )
            
            # 可选：为当前参数组合保存一张单独的回测图
            if save_plots and self.visualizer is not None:
                try:
                    orig_dir = self.visualizer.total_factor_file_dir
                    tag = f"sig{sig}_pt{pt_sl_list[0]}_sl{pt_sl_list[1]}_mh{max_h_list[0]}d{max_h_list[1]}h"
                    new_dir = os.path.join(orig_dir, plot_subdir, tag)
                    self.visualizer.total_factor_file_dir = new_dir
                    self.plot_results(model_name or 'Ensemble')
                finally:
                    self.visualizer.total_factor_file_dir = orig_dir
            
            if model_name not in self.backtest_results:
                print(f"  ⚠️ 模型 {model_name} 不在回测结果中，跳过此组合")
                continue
            
            metrics_dict = self.backtest_results[model_name].get(f"{data_key}_metrics", {})
            if metric_key not in metrics_dict:
                print(f"  ⚠️ 指标 {metric_key} 不在 {data_key}_metrics 中，实际 keys={list(metrics_dict.keys())}")
                continue
            
            score = float(metrics_dict[metric_key])
            print(f"  -> {data_key} {metric_key} = {score:.4f}")
            
            if (best_score is None) or (score > best_score):
                best_score = score
                best_params = {
                    'signal_strength_pct': float(sig),
                    'triple_barrier_pt_sl': pt_sl_list,
                    'triple_barrier_max_holding': max_h_list,
                }
                # 记录当前最佳的简单摘要
                best_summary = {
                    'metric': metric_key,
                    'data_range': data_key,
                    'score': best_score,
                    'params': best_params,
                }
        
        # 4) 还原原始配置，并用最佳参数再跑一遍，得到最终策略状态
        self.config = orig_config
        if best_params is not None:
            self.config['signal_strength_pct'] = best_params['signal_strength_pct']
            self.config['triple_barrier_pt_sl'] = best_params['triple_barrier_pt_sl']
            self.config['triple_barrier_max_holding'] = best_params['triple_barrier_max_holding']
            
            print("\n" + "=" * 60)
            print(f"参数搜索结束，最佳组合：{best_params} | {data_key} {metric_key}={best_score:.4f}")
            print("使用最佳参数重新运行完整策略流程，用于后续分析与保存...")
            print("=" * 60)
            
            self.run_full_pipeline(
                is_pre_load=is_pre_load,
                weight_method=weight_method,
                normalize_method=normalize_method,
                enable_factor_selection=enable_factor_selection,
            )
        else:
            print("⚠️ 参数搜索未找到任何有效组合，保持原始配置与结果。")
        
        return best_summary

