import numpy as np
import pandas as pd

# 大户的多头和空头总持仓量占比，大户指保证金余额排名前20%的用户。 
# 多仓持仓量比例 = 大户多仓持仓量 / 大户总持仓量 
# 空仓持仓量比例 = 大户空仓持仓量 / 大户总持仓量 
# 多空持仓量比值 = 多仓持仓量比例 / 空仓持仓量比例

# 大户持仓数据格式 (raw_whale_data): 
# { 
#     "open_time": "2025-10-01 00:00:00",  // 时间戳
#     "symbol": "ETHUSDT",                  // 交易对
#     "longAccount": "0.7353",              // 大户多仓持仓量比例
#     "longShortRatio": "2.7775",           // 大户多空持仓量比值
#     "shortAccount": "0.2647"              // 大户空仓持仓量比例
# }
class TopLongShortPositionRatioFactorEngine:
    
    def __init__(self, resample_freq='15m'):
        """
        聪明钱(Smart Money)因子挖掘引擎
        
        :param resample_freq: 数据重采样频率 (默认15分钟)
        
        注意：
        - raw_whale_data (在process方法中传入): 大户持仓数据，包含 ['open_time', 'symbol', 'longAccount', 'longShortRatio', 'shortAccount']
        """
        self.freq = resample_freq

    def process(self, raw_whale_data: pd.DataFrame, 
                windows=[24, 96],  # 滚动窗口 (如 24*15m=6h, 96*15m=24h)
                z_window=672):     # Z-Score 回看窗口 (如 7天)
        
        print(f"[*] 启动聪明钱引擎 | 频率: {self.freq}")
        
        # 1. 数据清洗与标准化 (Data Cleaning)
        # 解决重复数据问题，并转换数值类型
        df = self._clean_and_normalize(raw_whale_data)
        
        # 2. 基础特征构建 (Base Features)
        # 转换 Ratio 为 Net Position，计算一阶差分
        print("[-] 构建基础大户头寸特征...")
        df = self._build_base_features(df)
        
        # 3. 统计与时序特征 (Statistical & Temporal)
        # 计算 Z-Score (拥挤度) 和 ROC (变化率)
        print(f"[-] 计算统计特征 (Z-Window={z_window})...")
        df = self._build_stat_features(df, z_window)
        
        # 4. 持仓极值特征 (Position Extremes)
        # 核心逻辑：基于大户持仓的局部极值识别潜在信号
        print(f"[-] 挖掘大户持仓极值因子 (Windows={windows})...")
        df = self._build_position_signals(df, windows)
        
        print(f"[+] 处理完成. 最终数据形状: {df.shape}")
        return df

    def _clean_and_normalize(self, df: pd.DataFrame):
        """
        清洗阶段：去重、类型转换、时间排序
        """
        # 1. 复制数据避免修改原始数据
        df = df.copy()
        
        # 2. 类型强制转换
        df['open_time'] = pd.to_datetime(df['open_time'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
        df['longShortRatio'] = df['longShortRatio'].astype(float)
        df['longAccount'] = df['longAccount'].astype(float)
        df['shortAccount'] = df['shortAccount'].astype(float)
        
        # 3. 针对截图中的重复数据进行去重，保留最新的一条
        df = df.drop_duplicates(subset=["open_time", "symbol"], keep="last")
        
        # 4. 排序
        df = df.sort_values("open_time", ascending=True)
        
        # 5. 按频率重采样 (防止数据过于密集或稀疏)
        df = df.set_index('open_time')
        df = df.resample(self.freq).agg({
            'longShortRatio': 'last',
            'longAccount': 'last',
            'shortAccount': 'last'
        }).dropna()
        df = df.reset_index()
        
        return df

    def _build_base_features(self, df: pd.DataFrame):
        """
        将原始 Ratio 转化为更有物理意义的 Net Position
        """
        df = df.copy()
        
        # 1. 净头寸 (Net Position): 范围 [-1, 1]
        # 正数代表大户净看多，负数代表净看空
        df['whale_net_pos'] = df['longAccount'] - df['shortAccount']
        
        # 2. 净头寸流向 (Flow): 大户是在加仓还是减仓？
        df['whale_net_flow'] = (df['longAccount'] - df['shortAccount']).diff()
        
        return df

    def _build_stat_features(self, df: pd.DataFrame, window_size):
        """
        计算 Z-Score 以衡量"拥挤度"和"极端情绪"
        """
        df = df.copy()
        
        # 辅助函数：计算 Z-Score
        def z_score(series, w):
            rolling_mean = series.rolling(window=w, min_periods=1).mean()
            rolling_std = series.rolling(window=w, min_periods=1).std()
            return (series - rolling_mean) / (rolling_std + 1e-9)
        
        # 头寸的极端程度
        df[f'feat_z_whale_net_pos_{window_size}'] = z_score(df['whale_net_pos'], window_size)
        
        # 比率的极端程度
        df[f'feat_z_longShortRatio_{window_size}'] = z_score(df['longShortRatio'], window_size)
        
        return df
    
    def _safe_normalize(self, series: pd.Series, window, min_range=1e-6):
        """
        安全的归一化方法，处理边界情况
        
        :param series: Pandas Series
        :param window: 滚动窗口大小
        :param min_range: 最小有效范围，低于此值视为无变化
        :return: 归一化后的 Series (0~1)
        """
        min_val = series.rolling(window=window, min_periods=1).min()
        max_val = series.rolling(window=window, min_periods=1).max()
        range_val = max_val - min_val
        
        # 当范围过小时（价格/持仓无变化），返回中性值 0.5
        # 避免除零和数值不稳定
        normalized = np.where(
            range_val > min_range,
            (series - min_val) / range_val,
            0.5
        )
        
        return pd.Series(normalized, index=series.index)

    def _build_position_signals(self, df: pd.DataFrame, windows, 
                               high_thresh=0.8, 
                               low_thresh=0.2):
        """
        【核心逻辑】基于大户持仓的局部极值识别信号
        方法论：归一化大户持仓，识别高位和低位
        
        :param windows: 滚动窗口列表
        :param high_thresh: 高位阈值 (默认 0.8，即 80分位)
        :param low_thresh: 低位阈值 (默认 0.2，即 20分位)
        """
        df = df.copy()
        
        for w in windows:
            # 1. 大户净头寸的局部位置 (0~1) - 使用安全归一化
            whale_norm = self._safe_normalize(df['whale_net_pos'], w)
            
            # 2. 大户多空比的局部位置 (0~1) - 使用安全归一化
            ratio_norm = self._safe_normalize(df['longShortRatio'], w)
            
            # 3. 归一化后的净头寸 (可直接用作因子)
            df[f'feat_whale_pos_norm_w{w}'] = whale_norm
            
            # 4. 归一化后的多空比 (可直接用作因子)
            df[f'feat_ratio_norm_w{w}'] = ratio_norm
            
            # 5. 做多情绪极端信号 (Bullish Extreme)
            # 定义：大户净头寸在高位 => 大户集体看多
            df[f'sig_bullish_extreme_w{w}'] = (whale_norm > high_thresh).astype(np.int8)
            
            # 6. 做空情绪极端信号 (Bearish Extreme)
            # 定义：大户净头寸在低位 => 大户集体看空
            df[f'sig_bearish_extreme_w{w}'] = (whale_norm < low_thresh).astype(np.int8)
            
            # 7. 多空比极端信号 (Ratio Extreme)
            # 定义：多空比在高位 => 多头持仓极度拥挤
            df[f'sig_ratio_high_w{w}'] = (ratio_norm > high_thresh).astype(np.int8)
            df[f'sig_ratio_low_w{w}'] = (ratio_norm < low_thresh).astype(np.int8)

        return df