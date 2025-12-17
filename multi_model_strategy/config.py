"""
配置管理模块
统一管理数据配置和策略配置
"""


class StrategyConfig:
    """策略配置类"""
    
    @staticmethod
    def get_default_config():
        """获取默认策略配置"""
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
            
            # 三层结构相关开关
            'enable_regime_layer': False,  # 是否启用 Regime / 环境层缩放
            'enable_risk_layer': False,    # 是否启用 风控 & 拥挤度层缩放
            
            # Regime / Risk 层使用的特征列
            'regime_trend_cols': None,    # 例如 ['regime_trend_96', 'trend_slope_96']
            'regime_vol_cols': None,      # 例如 ['regime_vol_24']
            'risk_crowding_cols': None,   # 例如 ['oi_zscore_24', 'toptrader_oi_skew_abs']
            'risk_impact_cols': None,     # 例如 ['amihud_illiq_20', 'gap_strength_14']
            'risk_funding_cols': None,    # 例如 ['funding_zscore_24h']
            
            # Triple Barrier 集成相关
            'use_triple_barrier_label': None,   # 是否用 TB 收益替代固定周期收益做回归标签
            'triple_barrier_pt_sl': [2, 2],      # [止盈倍数, 止损倍数]
            'triple_barrier_max_holding': [0, 12],# [天, 小时] 最大持仓时间
            # Lopez CUSUM 事件相关参数（用于 Triple Barrier 事件选择）
            'tb_cusum_h': 3.0,                  # CUSUM 阈值系数，越大事件越少
            'tb_min_events': 500,               # CUSUM 最少事件数，低于该值退回使用所有 bar
            
            # Kelly bet size 模式
            'use_kelly_bet_sizing': None,        # 是否使用基于 p、R 的 Kelly 仓位 sizing
            'kelly_fraction': 0.25,              # Fractional Kelly 系数 c（通常 0.1~0.5）
            
            # Base 模型信号离散化（enter 过滤）相关
            # 使用训练集预测绝对值的分位数作为阈值，仅在“强信号” bar 上持仓
            # 例如 70 表示使用 |pred_train| 的 70% 分位数作为阈值
            # 设为 None 可关闭该离散化逻辑，保持连续仓位
            'signal_strength_pct': 70,

            # 参数搜索（网格搜索）相关配置
            # 是否在单独调用 run_param_search 时启用这些默认网格
            # 注意：不会自动在 run_full_pipeline 中触发，仅在你显式调用时使用
            'param_search_signal_strength_grid': [50.0, 60.0, 70.0, 80.0, 90.0],
            # pt_sl 网格，元素为 [pt, sl]
            'param_search_triple_barrier_pt_sl_grid': [[1.0, 2.0], [2.0, 2.0], [2.0, 3.0]],
            # 最大持仓时间网格，元素为 [days, hours]
            'param_search_triple_barrier_max_holding_grid': [[0, 4], [0, 8], [0, 12]],
            # 评价指标与范围
            # metric 可选：'Sharpe Ratio', 'Calmar Ratio', 'Annual Return'
            'param_search_metric': 'Sharpe Ratio',
            # 在 train/test 哪一段上评估超参数
            'param_search_data_range': 'test',
            # 使用哪个模型名称做评估，默认 Ensemble
            'param_search_model_name': 'Ensemble',
        }
    
    @staticmethod
    def merge_config(base_config, user_config):
        """合并配置（用户配置覆盖默认配置）"""
        merged = base_config.copy()
        if user_config:
            merged.update(user_config)
        return merged


class DataConfig:
    """数据配置类"""
    
    @staticmethod
    def build_from_yaml(yaml_config):
        """从YAML配置构建数据配置"""
        return {
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
            'timeframe': yaml_config.get('timeframe', None),
            # coarse_grain 相关
            'coarse_grain_period': yaml_config.get('coarse_grain_period', '2h'),
            'feature_lookback_bars': yaml_config.get('feature_lookback_bars', 8),
            'rolling_step': yaml_config.get('rolling_step', '15min'),
            'file_path': yaml_config.get('file_path', None),
        }
    
    @staticmethod
    def build_simple(sym='ETHUSDT', freq='15m', 
                    train_dates=('2025-01-01', '2025-03-01'),
                    test_dates=('2025-03-01', '2025-04-01'),
                    **kwargs):
        """构建简化配置"""
        return {
            'sym': sym,
            'freq': freq,
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
