"""
主策略类
整合所有模块，提供统一的接口
"""
import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import joblib
import pickle

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

# Triple Barrier
from gp_crypto_next.triple_barrier import get_barrier, get_metalabel


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
        config['return_period'] = yaml_config.get('y_train_ret_period', 1)
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
    
    def run_full_pipeline(self, weight_method='equal', normalize_method=None,
                         enable_factor_selection=False):
        """
        运行完整策略流程
        
        Args:
            weight_method (str): 'equal' 或 'sharpe'
            normalize_method (str or None): 因子标准化方法（None=不标准化）
            enable_factor_selection (bool): 是否启用因子筛选
        """
        print("="*60)
        print("开始运行完整的量化策略流程（整合GP因子）")
        print("="*60)
        
        # 1. 数据加载
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
        
        # 生成 TB 标签
        close_series = pd.Series(
            data=self.data_module.ohlc[:, 3],
            index=pd.to_datetime(self.z_index)
        )
        
        rolling_window = self.data_config.get('rolling_window', 2000)
        target_volatility = close_series.pct_change().rolling(
            window=min(rolling_window, len(close_series)//2)
        ).std()
        target_volatility = target_volatility.fillna(method='bfill')
        
        self.barrier_results = get_barrier(
            close=close_series,
            enter=close_series.index,
            pt_sl=pt_sl,
            max_holding=max_holding,
            target=target_volatility,
            side=pd.Series(1.0, index=close_series.index)
        )
        
        self.meta_labels = get_metalabel(self.barrier_results)
        
        # 替换 label
        barrier_ret = self.barrier_results['ret'].values
        train_len = len(self.y_train)
        self.y_train = barrier_ret[:train_len].reshape(-1, 1)
        self.y_test = barrier_ret[train_len:train_len+len(self.y_test)].reshape(-1, 1)
        self.ret_train = self.y_train.flatten()
        self.ret_test = self.y_test.flatten()
        
        print(f"Triple Barrier 标签应用完成")
    
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
            self.data_module.open_train, self.data_module.close_train,
            self.data_module.open_test, self.data_module.close_test,
            self.config['fees_rate'], self.config['annual_bars']
        )
        
        self.alpha_trainer = AlphaModelTrainer(
            X_train, X_test, self.y_train, self.y_test, selected_factors
        )
        
        self.alpha_trainer.train_all_models(use_normalized_label=True)
        self.alpha_trainer.make_predictions()
        
        # 模型集成
        backtest_fn = lambda pos, data_range: self.backtest_engine.run_backtest(pos, data_range)
        self.alpha_trainer.ensemble_models(weight_method, backtest_fn)
        
        self.predictions = self.alpha_trainer.get_predictions()
        self.ensemble_weights = self.alpha_trainer.ensemble_weights
    
    def _apply_kelly_sizing(self):
        """步骤6（可选）：Kelly bet sizing"""
        print("ℹ️ 启用 Lopez 风格 Kelly bet sizing")
        
        if self.barrier_results is None or self.meta_labels is None:
            print("⚠️ Kelly 需要 Triple Barrier，自动生成...")
            self._apply_triple_barrier_labels()
        
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

