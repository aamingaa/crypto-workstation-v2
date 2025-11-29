"""
数据加载模块
封装 dataload 调用逻辑
"""
import sys
from pathlib import Path

# 确保 gp_crypto_next 在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import gp_crypto_next.dataload as dataload


class DataModule:
    """
    数据模块：负责调用 dataload 加载数据
    """
    
    def __init__(self, data_config, strategy_config):
        """
        Args:
            data_config (dict): 数据配置（sym, freq, dates等）
            strategy_config (dict): 策略配置（return_period等）
        """
        self.data_config = data_config
        self.strategy_config = strategy_config
        
        # 数据容器
        self.X_all = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.ret_train = None
        self.ret_test = None
        self.feature_names = None
        self.z_index = None
        self.ohlc = None
        self.open_train = None
        self.open_test = None
        self.close_train = None
        self.close_test = None
        self.y_p_train_origin = None
        self.y_p_test_origin = None
    
    def load(self):
        """加载数据"""
        print("正在使用dataload模块加载数据...")
        
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
        data_source = self.data_config.get('data_source', 'kline')
        
        try:
            if str(data_source).lower() == 'coarse_grain':
                print(f"使用粗粒度特征方法 (coarse_grain)")
                coarse_grain_period = self.data_config.get('coarse_grain_period', '2h')
                feature_lookback_bars = self.data_config.get('feature_lookback_bars', 8)
                rolling_step = self.data_config.get('rolling_step', '15min')
                file_path = self.data_config.get('file_path', None)
                
                (self.X_all, self.X_train, self.y_train, self.ret_train,
                 self.X_test, self.y_test, self.ret_test, self.feature_names,
                 self.open_train, self.open_test, self.close_train, self.close_test,
                 self.z_index, self.ohlc, self.y_p_train_origin, self.y_p_test_origin
                 ) = dataload.data_prepare_coarse_grain_rolling(
                    sym, freq, start_date_train, end_date_train,
                    start_date_test, end_date_test,
                    coarse_grain_period=coarse_grain_period,
                    feature_lookback_bars=feature_lookback_bars,
                    rolling_step=rolling_step,
                    y_train_ret_period=self.strategy_config['return_period'],
                    rolling_w=rolling_w,
                    output_format='ndarry',
                    data_dir=data_dir,
                    read_frequency=read_frequency,
                    timeframe=timeframe,
                    file_path=file_path,
                    include_categories=['momentum']
                )
                
            elif str(data_source).lower() == 'kline':
                print(f"使用标准 K线 数据方法 (kline)")
                (self.X_all, self.X_train, self.y_train, self.ret_train,
                 self.X_test, self.y_test, self.ret_test, self.feature_names,
                 self.open_train, self.open_test, self.close_train, self.close_test,
                 self.z_index, self.ohlc) = dataload.data_prepare(
                    sym, freq, start_date_train, end_date_train,
                    start_date_test, end_date_test,
                    y_train_ret_period=self.strategy_config['return_period'],
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
            
            return self
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            raise
    
    def get_data_dict(self):
        """返回数据字典（方便传递给其他模块）"""
        return {
            'X_all': self.X_all,
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'ret_train': self.ret_train,
            'ret_test': self.ret_test,
            'feature_names': self.feature_names,
            'z_index': self.z_index,
            'ohlc': self.ohlc,
            'open_train': self.open_train,
            'open_test': self.open_test,
            'close_train': self.close_train,
            'close_test': self.close_test,
            'y_p_train_origin': self.y_p_train_origin,
            'y_p_test_origin': self.y_p_test_origin,
        }

