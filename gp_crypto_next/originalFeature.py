"""
加工原始特征，衍生新的特征
"""
import datetime
from dataclasses import dataclass, field
from typing import List
import pandas as pd
import numpy as np
import talib
from scipy.stats import rankdata
from functools import singledispatch
import warnings
import dataload
import yaml
from tqdm import tqdm
from .LiquidationFeature import get_advanced_liquidation_features

warnings.filterwarnings('ignore')


def define_base_fields(rolling_zscore_window: int = 2000, include_categories: List[str] = None, init_ohlcva_df: pd.DataFrame = None):
    """
    本函数定义了基础的特征计算公式. 将来会持续维护这个函数，增加更多的特征计算公式
    这是唯一的定义特征的地方，其他地方不应该再定义特征
    """
    def fm20(c: pd.Series, o: pd.Series) -> pd.Series:
        '''输入：close，open
        输出：窗口为20的波动的周期特征fm20,from John Elther
        '''
        period = 20
        Deriv = c - o
        # 滤除AM噪音
        HL = np.clip(10 * Deriv, -1, 1)
        a1 = np.exp(-1.414 * np.pi / period)
        b1 = 2 * a1 * np.cos(1.414 * 180 / period)
        # c2 = b1
        c3 = -a1 * a1
        c1 = 1 - b1 - c3
        # 初始化输出序列
        ss = np.zeros_like(HL)
        ss[:2] = HL[:2]  # 使用前两个值作为初始化值
        for i in range(2, len(HL)):
            ss[i] = c1 * (HL[i] + HL[i-1]) / 2 + b1 * ss[i-1] + c3 * ss[i-2]
        return ss

    def fm30(c: pd.Series, o: pd.Series) -> pd.Series:
        '''输入：close，open
        输出：窗口为30的波动的周期特征fm30,from John Elther
        '''
        period = 30
        Deriv = c - o
        # 滤除AM噪音
        HL = np.clip(10 * Deriv, -1, 1)
        a1 = np.exp(-1.414 * np.pi / period)
        b1 = 2 * a1 * np.cos(1.414 * 180 / period)
        # c2 = b1
        c3 = -a1 * a1
        c1 = 1 - b1 - c3
        # 初始化输出序列
        ss = np.zeros_like(HL)
        ss[:2] = HL[:2]  # 使用前两个值作为初始化值
        for i in range(2, len(HL)):
            ss[i] = c1 * (HL[i] + HL[i-1]) / 2 + b1 * ss[i-1] + c3 * ss[i-2]
        return ss

    def fm40(c: pd.Series, o: pd.Series) -> pd.Series:
        '''输入：close，open
        输出：窗口为40的波动的周期特征fm40,from John Elther
        '''
        period = 40
        Deriv = c - o
        # 滤除AM噪音
        HL = np.clip(10 * Deriv, -1, 1)
        a1 = np.exp(-1.414 * np.pi / period)
        b1 = 2 * a1 * np.cos(1.414 * 180 / period)
        # c2 = b1
        c3 = -a1 * a1
        c1 = 1 - b1 - c3
        # 初始化输出序列
        ss = np.zeros_like(HL)
        ss[:2] = HL[:2]  # 使用前两个值作为初始化值
        for i in range(2, len(HL)):
            ss[i] = c1 * (HL[i] + HL[i-1]) / 2 + b1 * ss[i-1] + c3 * ss[i-2]
        return ss

    def fm60(c: pd.Series, o: pd.Series) -> pd.Series:
        '''输入：close，open
        输出：窗口为60的波动的周期特征fm60,from John Elther
        '''
        period = 60
        Deriv = c - o
        # 滤除AM噪音
        HL = np.clip(10 * Deriv, -1, 1)
        a1 = np.exp(-1.414 * np.pi / period)
        b1 = 2 * a1 * np.cos(1.414 * 180 / period)
        # c2 = b1
        c3 = -a1 * a1
        c1 = 1 - b1 - c3
        # 初始化输出序列
        ss = np.zeros_like(HL)
        ss[:2] = HL[:2]  # 使用前两个值作为初始化值
        for i in range(2, len(HL)):
            ss[i] = c1 * (HL[i] + HL[i-1]) / 2 + b1 * ss[i-1] + c3 * ss[i-2]
        return ss

    def norm(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        # mean = pd.Series(x).rolling(2000, min_periods=1).mean().values
        std = pd.Series(x).rolling(rolling_zscore_window, min_periods=1).std().values
        # x_value = (x - mean) / np.clip(np.nan_to_num(std),
        #                                a_min=1e-6, a_max=None)
        x_value = (x ) / np.clip(np.nan_to_num(std),
                                       a_min=1e-6, a_max=None)
        # x_value = np.clip(x_value, -6, 6)
        x_value = np.nan_to_num(x_value, nan=0.0, posinf=0.0, neginf=0.0)
        return x_value

    # ---- 辅助函数（仅本作用域内使用）----
    def _safe_div(numer: np.ndarray, denom: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """数值安全除法，避免 0 除与极端值。"""
        return np.asarray(numer, dtype=np.float64) / np.maximum(np.asarray(denom, dtype=np.float64), eps)

    def parkinson_var(h: np.ndarray, l: np.ndarray) -> np.ndarray:
        """Parkinson 波动估计，仅用高低价，适合无成交价路径的小时线。"""
        return (1.0 / (4.0 * np.log(2.0))) * (np.log(_safe_div(h, l)) ** 2)

    def garman_klass_var(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Garman–Klass 波动估计，结合高低与开收，排除漂移影响。"""
        return 0.5 * (np.log(_safe_div(h, l)) ** 2) - (2.0 * np.log(2.0) - 1.0) * (np.log(_safe_div(c, o)) ** 2)

    def rogers_satchell_var(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Rogers–Satchell 波动估计，对漂移更鲁棒。"""
        return np.log(_safe_div(h, c)) * np.log(_safe_div(h, o)) + np.log(_safe_div(l, c)) * np.log(_safe_div(l, o))

    def clv(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Close Location Value：收盘在当根区间的位置，[-1,1]。"""
        rng = np.maximum(h - l, 1e-12)
        return ((c - l) - (h - c)) / rng

    def body_upper_lower(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray):
        """返回实体比例、上影比例、下影比例，均以全区间 H-L 归一。"""
        rng = np.maximum(h - l, 1e-12)
        body = np.abs(c - o) / rng
        upper = (h - np.maximum(o, c)) / rng
        lower = (np.minimum(o, c) - l) / rng
        return body, upper, lower
    
    features = {
        'liq_zscore': lambda data: calc_liquidation_pressure(data, lookback=24),
        'lgp_shortcut_illiq_6': lambda data: norm(np.nan_to_num(pd.Series(2*(data['h'] - data['l']) - np.abs(data['c'] - data['o'])).rolling(6, min_periods=1).apply(lambda x: x.mean()))),
        'h_ts_std_10': lambda data: norm(np.nan_to_num(pd.Series(data['h']).rolling(window=10, min_periods=5).std())),
        'v_ta_cmo_25': lambda data: norm(talib.CMO(data['vol'], 25)),
        'v_ts_argrange_10': lambda data: norm(np.nan_to_num(pd.Series(data['vol']).rolling(window=10, min_periods=5).apply(np.argmax) + 1) - np.nan_to_num(pd.Series(data['vol']).rolling(window=10, min_periods=5).apply(np.argmin) + 1)),
        'c_ts_argrange_10': lambda data: norm(np.nan_to_num(pd.Series(data['c']).rolling(window=10, min_periods=5).apply(np.argmax) + 1) - np.nan_to_num(pd.Series(data['c']).rolling(window=10, min_periods=5).apply(np.argmin) + 1)),
        'c_ts_day_min_40': lambda data: norm(np.nan_to_num(pd.Series(data['c']).rolling(window=41).apply(lambda x: 40 - x.values.tolist()[:-1].index(np.min(x.values.tolist()[:-1]))))),
        'l_ts_prod_8': lambda data: norm(np.nan_to_num(pd.Series(data['l']).rolling(window=8, min_periods=4).apply(np.prod))),
        'v_ts_std_20': lambda data: norm(np.nan_to_num(pd.Series(data['vol']).rolling(window=20, min_periods=10).std())),
        'h_ta_lr_angle_10': lambda data: norm(np.nan_to_num(talib.LINEARREG_ANGLE(data['h'], timeperiod=10))),
        'v_ts_sma_21': lambda data: norm(np.nan_to_num(pd.Series(data['vol']).rolling(window=21, min_periods=11).mean())),
        'h_ts_kurt_20': lambda data: norm(np.nan_to_num(pd.Series(data['h']).rolling(window=20, min_periods=10).kurt())),
        'v_ts_range_5': lambda data: norm(np.nan_to_num(pd.Series(data['vol']).rolling(window=5, min_periods=3).max()) - np.nan_to_num(pd.Series(data['vol']).rolling(window=5, min_periods=3).min())),
        'c_ts_sma_21': lambda data: norm(np.nan_to_num(pd.Series(data['c']).rolling(window=21, min_periods=11).mean())),
        'o_ts_argmin_10': lambda data: norm(np.nan_to_num(pd.Series(data['o']).rolling(window=10, min_periods=5).apply(np.argmin) + 1)),
        'o_ta_lr_slope_10': lambda data: norm(np.nan_to_num(talib.LINEARREG_SLOPE(data['o'], timeperiod=10))),
        'l_ts_argmax_20': lambda data: norm(np.nan_to_num(pd.Series(data['l']).rolling(window=20, min_periods=10).apply(np.argmax) + 1)),
        'o_ts_range_10': lambda data: norm(np.nan_to_num(pd.Series(data['o']).rolling(window=10, min_periods=5).max()) - np.nan_to_num(pd.Series(data['o']).rolling(window=10, min_periods=5).min())),
        'h_ts_argmax_5': lambda data: norm(np.nan_to_num(pd.Series(data['h']).rolling(window=5, min_periods=3).apply(np.argmax) + 1)),
        'l_ts_day_min_40': lambda data: norm(np.nan_to_num(pd.Series(data['l']).rolling(window=41).apply(lambda x: 40 - x.values.tolist()[:-1].index(np.min(x.values.tolist()[:-1]))))),
        'v_ts_day_max_20': lambda data: norm(np.nan_to_num(pd.Series(data['vol']).rolling(window=21).apply(lambda x: 20 - x.values.tolist()[:-1].index(np.max(x.values.tolist()[:-1]))))),
        'c_ta_tsf_5': lambda data: norm(np.nan_to_num(talib.TSF(data['c'], timeperiod=5))),
        'c_power_c': lambda data: norm(np.nan_to_num(np.power(data['c'], 3))),
        'l_ts_argrange_5': lambda data: norm(np.nan_to_num(pd.Series(data['l']).rolling(window=5, min_periods=3).apply(np.argmax) + 1) - np.nan_to_num(pd.Series(data['l']).rolling(window=5, min_periods=3).apply(np.argmin) + 1)),
        'v_power_a': lambda data: norm(np.nan_to_num(np.power(data['vol'], 3))),
        'v_cci_25_obv': lambda data: norm(talib.OBV(norm(np.nan_to_num(talib.CCI(data['h'], data['l'], data['c'], timeperiod=25))), data['vol'])),
        'v_cci_25_sum': lambda data: norm(np.cumsum(norm(np.nan_to_num(talib.CCI(data['h'], data['l'], data['c'], timeperiod=25))) * data['vol'])),
        'ori_trix_8': lambda data: norm(np.nan_to_num(talib.TRIX(data['c'], timeperiod=8))),
        'ori_trix_21': lambda data: norm(np.nan_to_num(talib.TRIX(data['c'], timeperiod=21))),
        'ori_trix_55': lambda data: norm(np.nan_to_num(talib.TRIX(data['c'], timeperiod=55))),
        'v_trix_8_obv': lambda data: norm(talib.OBV(norm(np.nan_to_num(talib.TRIX(data['c'], timeperiod=8))), data['vol'])),
        'v_sar_obv': lambda data: norm(talib.OBV(norm(np.nan_to_num(talib.SAR(data['h'], data['l']))), data['vol'])),
        'ori_ta_bop': lambda data: norm(talib.BOP(data['o'], data['h'], data['l'], data['c'])),
        'v_bop_sum': lambda data: norm(np.cumsum(norm(talib.BOP(data['o'], data['h'], data['l'], data['c'])) * data['vol'])),
        'ori_rsi_6': lambda data: norm(talib.RSI(data['c'], timeperiod=6)),
        'ori_rsi_12': lambda data: norm(talib.RSI(data['c'], timeperiod=12)),
        'ori_rsi_24': lambda data: norm(talib.RSI(data['c'], timeperiod=24)),
        'v_rsi_6_obv': lambda data: norm(talib.OBV(norm(talib.RSI(data['c'], timeperiod=6)), data['vol'])),
        'v_rsi_6_sum': lambda data: norm(np.cumsum(norm(talib.RSI(data['c'], timeperiod=6)) * data['vol'])),
        'v_rsi_12_obv': lambda data: norm(talib.OBV(norm(talib.RSI(data['c'], timeperiod=12)), data['vol'])),
        'v_rsi_12_sum': lambda data: norm(np.cumsum(norm(talib.RSI(data['c'], timeperiod=12)) * data['vol'])),
        'v_rsi_24_obv': lambda data: norm(talib.OBV(norm(talib.RSI(data['c'], timeperiod=24)), data['vol'])),
        'v_rsi_24_sum': lambda data: norm(np.cumsum(norm(talib.RSI(data['c'], timeperiod=24)) * data['vol'])),
        'ori_cmo_14': lambda data: norm(np.nan_to_num(talib.CMO(data['c'], timeperiod=14))),
        'ori_cmo_25': lambda data: norm(np.nan_to_num(talib.CMO(data['c'], timeperiod=25))),
        'v_cmo_14_obv': lambda data: norm(talib.OBV(norm(np.nan_to_num(talib.CMO(data['c'], timeperiod=14))), data['vol'])),
        'fm20': lambda data: norm(np.nan_to_num(fm20(data['c'], data['o']))),
        'fm30': lambda data: norm(np.nan_to_num(fm30(data['c'], data['o']))),
        'fm40': lambda data: norm(np.nan_to_num(fm40(data['c'], data['o']))),
        'fm60': lambda data: norm(np.nan_to_num(fm60(data['c'], data['o']))),
        'ori_ta_macd': lambda data: norm(norm(np.nan_to_num(talib.MACD(data['c'], fastperiod=12, slowperiod=26, signalperiod=9)[2]))),
        'ori_ta_obv': lambda data: norm(np.nan_to_num(talib.OBV(data['c'], data['vol']))),
        'ori_ta_ad': lambda data: norm(np.nan_to_num(talib.AD(data['h'], data['l'], data['c'], data['vol']))),
        'ori_ta_bop': lambda data: norm(talib.BOP(data['o'], data['h'], data['l'], data['c'])),
        'ma8_c': lambda data: norm(np.nan_to_num(talib.MA(data['c'], timeperiod=8, matype=0) / data['c'])),
        'ma15_c': lambda data: norm(np.nan_to_num(talib.MA(data['c'], timeperiod=15, matype=0) / data['c'])),
        'ma25_c': lambda data: norm(np.nan_to_num(talib.MA(data['c'], timeperiod=25, matype=0) / data['c'])),
        'ma35_c': lambda data: norm(np.nan_to_num(talib.MA(data['c'], timeperiod=35, matype=0) / data['c'])),
        'ma55_c': lambda data: norm(np.nan_to_num(talib.MA(data['c'], timeperiod=55, matype=0) / data['c'])),
        'h_line': lambda data: norm(np.nan_to_num(talib.BBANDS(data['c'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0])),
        'm_line': lambda data: norm(np.nan_to_num(talib.BBANDS(data['c'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1])),
        'l_line': lambda data: norm(np.nan_to_num(talib.BBANDS(data['c'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2])),
        'stdevrate': lambda data: norm((talib.BBANDS(data['c'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0] -
                                        talib.BBANDS(data['c'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2]) /
                                       (data['c'] * 4)),
        'sar_index': lambda data: norm(np.nan_to_num(talib.SAR(data['h'], data['l']))),
        'sar_close': lambda data: norm((np.nan_to_num(talib.SAR(data['h'], data['l'])) - data['c']) / data['c']),
        'mfi_index': lambda data: norm(np.nan_to_num(talib.MFI(data['h'], data['l'], data['c'], data['vol']))),
        'mfi_30': lambda data: norm(np.nan_to_num(talib.MFI(data['h'], data['l'], data['c'], data['vol'], timeperiod=30))),
        'ppo': lambda data: norm(np.nan_to_num(talib.PPO(data['c'], fastperiod=12, slowperiod=26, matype=0))),
        'ad_index': lambda data: norm(np.nan_to_num(talib.AD(data['h'], data['l'], data['c'], data['vol']))),
        'ad_real': lambda data: norm(np.nan_to_num(talib.ADOSC(data['h'], data['l'], data['c'], data['vol'], fastperiod=3, slowperiod=10))),
        'tr_index': lambda data: norm(np.nan_to_num(talib.TRANGE(data['h'], data['l'], data['c']))),
        'sarext': lambda data: norm(np.nan_to_num(talib.SAREXT(data['h'], data['l'], startvalue=0, offsetonreverse=0,
                                                               accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0,
                                                               accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0))),
        'kdj_d': lambda data: talib.STOCH(data['h'], data['l'], data['c'], 9, 3, 3)[1],
        'kdj_k': lambda data: talib.STOCH(data['h'], data['l'], data['c'], 9, 3, 3)[0],
        'obv_v': lambda data: norm(np.nan_to_num(talib.OBV(data['c'], data['vol']))),
        'volume_macd': lambda data: norm(np.nan_to_num(talib.MACD(data['vol'], fastperiod=12, slowperiod=26, signalperiod=9)[2])),
        'close_macd': lambda data: norm(np.nan_to_num(talib.MACD(data['c'], fastperiod=12, slowperiod=26, signalperiod=9)[2])),
        'cci_55': lambda data: norm(talib.CCI(data['c'], data['l'], data['h'], 55)),
        'cci_25': lambda data: norm(talib.CCI(data['c'], data['l'], data['h'], 25)),
        'cci_14': lambda data: norm(talib.CCI(data['c'], data['l'], data['h'], 14)),

        # ---------------- 下面为新增：波动率估计（高低开收修正） ----------------
        'var_parkinson': lambda data: norm(np.nan_to_num(parkinson_var(data['h'], data['l']))),
        'var_garman_klass': lambda data: norm(np.nan_to_num(garman_klass_var(data['o'], data['h'], data['l'], data['c']))),
        'var_rogers_satchell': lambda data: norm(np.nan_to_num(rogers_satchell_var(data['o'], data['h'], data['l'], data['c']))),

        # ---------------- ATR 家族与压缩/扩张 ----------------
        'atr_14': lambda data: norm(np.nan_to_num(talib.ATR(data['h'], data['l'], data['c'], timeperiod=14))),
        'range_over_atr_14': lambda data: norm(_safe_div(data['h'] - data['l'], np.nan_to_num(talib.ATR(data['h'], data['l'], data['c'], timeperiod=14)) + 1e-12)),
        'atr_over_maatr_14_50': lambda data: norm(_safe_div(
            np.nan_to_num(talib.ATR(data['h'], data['l'], data['c'], timeperiod=14)),
            np.nan_to_num(talib.SMA(talib.ATR(data['h'], data['l'], data['c'], timeperiod=14), timeperiod=50)) + 1e-12
        )),
        'bb_width_over_atr_20': lambda data: (
            lambda ub, mb, lb, atr: norm(_safe_div(np.nan_to_num(ub - lb), atr + 1e-12))
        )(*talib.BBANDS(data['c'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0), talib.ATR(data['h'], data['l'], data['c'], timeperiod=14)),

        # ---------------- K 线结构/位置 ----------------
        'clv': lambda data: norm(np.nan_to_num(clv(data['o'], data['h'], data['l'], data['c']))),
        'body_ratio': lambda data: (
            lambda body, upper, lower: norm(np.nan_to_num(body))
        )(*body_upper_lower(data['o'], data['h'], data['l'], data['c'])),
        'upper_shadow_ratio': lambda data: (
            lambda body, upper, lower: norm(np.nan_to_num(upper))
        )(*body_upper_lower(data['o'], data['h'], data['l'], data['c'])),
        'lower_shadow_ratio': lambda data: (
            lambda body, upper, lower: norm(np.nan_to_num(lower))
        )(*body_upper_lower(data['o'], data['h'], data['l'], data['c'])),
        'donchian_pos_20': lambda data: (
            lambda ll, hh: norm(_safe_div(data['c'] - ll, hh - ll + 1e-12))
        )(pd.Series(data['l']).rolling(window=20, min_periods=5).min(),
          pd.Series(data['h']).rolling(window=20, min_periods=5).max()),

        # ---------------- 缺口/冲击 ----------------
        'gap_strength_14': lambda data: (
            lambda atr14, cp: norm(_safe_div(np.abs(data['o'] - cp), atr14 + 1e-12))
        )(np.nan_to_num(talib.ATR(data['h'], data['l'], data['c'], timeperiod=14)), pd.Series(data['c']).shift(1).values),
        'amihud_illiq_20': lambda data: norm(np.nan_to_num(_safe_div(
            np.abs(pd.Series(data['c']).pct_change().values),
            pd.Series(data['vol']).rolling(window=20, min_periods=10).mean().replace(0, np.nan).values
        ))),

        # ---------------- 价量关系 ----------------
        'ret_vol_corr_20': lambda data: (
            lambda corr: norm(np.nan_to_num(corr))
        )(pd.Series(pd.Series(data['c']).pct_change()).rolling(window=20, min_periods=10).corr(pd.Series(data['vol']))),
        'obv_lr_slope_20': lambda data: norm(np.nan_to_num(talib.LINEARREG_SLOPE(talib.OBV(data['c'], data['vol']), timeperiod=20))),

        # =====================================================================
        # 下面为新增因子：更结构化的趋势/动量 + 价量关系 + 冲击/流动性
        # =====================================================================

        # ---------------- 趋势 / 动量：多周期收益与趋势斜率 ----------------
        # 多周期 log 收益（使用 pct_change 近似），均用 rolling_zscore_window 做标准化
        'ret_1': lambda data: norm(np.nan_to_num(pd.Series(data['c']).pct_change(1).values)),
        'ret_4': lambda data: norm(np.nan_to_num(pd.Series(data['c']).pct_change(4).values)),
        'ret_12': lambda data: norm(np.nan_to_num(pd.Series(data['c']).pct_change(12).values)),
        'ret_24': lambda data: norm(np.nan_to_num(pd.Series(data['c']).pct_change(24).values)),
        'ret_48': lambda data: norm(np.nan_to_num(pd.Series(data['c']).pct_change(48).values)),
        'ret_96': lambda data: norm(np.nan_to_num(pd.Series(data['c']).pct_change(96).values)),

        # 基于 log(close) 的趋势斜率（短/中周期）
        'trend_slope_24': lambda data: norm(np.nan_to_num(
            talib.LINEARREG_SLOPE(np.log(np.maximum(data['c'], 1e-12)), timeperiod=24)
        )),
        'trend_slope_72': lambda data: norm(np.nan_to_num(
            talib.LINEARREG_SLOPE(np.log(np.maximum(data['c'], 1e-12)), timeperiod=72)
        )),
        'trend_slope_168': lambda data: norm(np.nan_to_num(
            talib.LINEARREG_SLOPE(np.log(np.maximum(data['c'], 1e-12)), timeperiod=168)
        )),

        # 趋势一致性：过去 24 根上涨比例
        'up_ratio_24': lambda data: norm(np.nan_to_num(
            pd.Series(data['c']).pct_change().rolling(window=24, min_periods=12).apply(
                lambda x: np.mean(x > 0)
            ).values
        )),

        # 更长周期的 Donchian 通道位置（长期趋势/位置）
        'donchian_pos_50': lambda data: (
            lambda ll, hh: norm(_safe_div(data['c'] - ll, hh - ll + 1e-12))
        )(pd.Series(data['l']).rolling(window=50, min_periods=10).min(),
          pd.Series(data['h']).rolling(window=50, min_periods=10).max()),
        'donchian_pos_200': lambda data: (
            lambda ll, hh: norm(_safe_div(data['c'] - ll, hh - ll + 1e-12))
        )(pd.Series(data['l']).rolling(window=200, min_periods=40).min(),
          pd.Series(data['h']).rolling(window=200, min_periods=40).max()),

        # ---------------- 价量关系：量能惊喜 & 成交结构 ----------------
        # 量能水平（合约张数、计价货币）
        'vol_level': lambda data: norm(np.nan_to_num(data.get('vol', 0.0))),
        'vol_ccy_level': lambda data: norm(np.nan_to_num(data.get('vol_ccy', 0.0))),

        # 平均单笔成交量及其标准化：大单主导 vs 零散成交
        'avg_trade_size': lambda data: norm(np.nan_to_num(
            _safe_div(data.get('vol', 0.0), np.maximum(data.get('trades', 0.0), 1.0))
        )),

        # 24 根内基于持仓的“换手率”近似：高换手≃筹码快速轮动
        'turnover_oi_24': lambda data: norm(np.nan_to_num(_safe_div(
            pd.Series(data.get('vol', 0.0)).rolling(window=24, min_periods=12).sum().values,
            np.maximum(data.get('oi', 0.0), 1e-12)
        ))),

        # ---------------- 冲击 / 流动性：多窗口 Amihud + 方向性成交 ----------------
        # 更短/更长窗口的 Amihud 式流动性指标
        'amihud_illiq_5': lambda data: norm(np.nan_to_num(_safe_div(
            np.abs(pd.Series(data['c']).pct_change().values),
            pd.Series(data['vol']).rolling(window=5, min_periods=3).mean().replace(0, np.nan).values
        ))),
        'amihud_illiq_60': lambda data: norm(np.nan_to_num(_safe_div(
            np.abs(pd.Series(data['c']).pct_change().values),
            pd.Series(data['vol']).rolling(window=60, min_periods=30).mean().replace(0, np.nan).values
        ))),

        # 带符号的缺口强度（相对 ATR14）：顺势/逆势 gap
        'gap_signed_14': lambda data: (
            lambda atr14, cp: norm(_safe_div((data['o'] - cp), atr14 + 1e-12))
        )(np.nan_to_num(talib.ATR(data['h'], data['l'], data['c'], timeperiod=14)),
          pd.Series(data['c']).shift(1).values),

        # 方向性主动成交不平衡（多空偏向），使用 taker_vol_lsr
        'taker_imbalance': lambda data: norm(np.nan_to_num(data.get('taker_vol_lsr', 0.0))),

        # =====================================================================
        # 拥挤度 & 杠杆风险相关因子（仅使用当前已知字段：oi / toptrader_* / taker_vol_lsr）
        # =====================================================================

        # OI 在短期窗口（24 根）内的 z-score（杠杆热度）
        'oi_zscore_24': lambda data: norm(np.nan_to_num(
            _safe_div(
                pd.Series(data.get('oi', 0.0)) - pd.Series(data.get('oi', 0.0)).rolling(window=24, min_periods=12).mean(),
                pd.Series(data.get('oi', 0.0)).rolling(window=24, min_periods=12).std().replace(0, np.nan)
            ).values
        )),

        # OI 24 根变化率（杠杆加速/减速）
        'oi_change_24': lambda data: norm(np.nan_to_num(
            _safe_div(
                pd.Series(data.get('oi', 0.0)).diff(24).values,
                pd.Series(data.get('oi', 0.0)).shift(24).replace(0, np.nan).values
            )
        )),

        # 大户持仓多空偏向（long/short ratio），直接标准化
        'toptrader_oi_skew': lambda data: norm(np.nan_to_num(data.get('toptrader_oi_lsr', 0.0))),
        'toptrader_count_skew': lambda data: norm(np.nan_to_num(data.get('toptrader_count_lsr', 0.0))),
        'toptrader_oi_skew_abs': lambda data: norm(np.nan_to_num(np.abs(data.get('toptrader_oi_lsr', 0.0)))),

        # 结合方向性主动成交与量能：大幅方向性交易 + 放量
        'taker_imbalance_vol': lambda data: norm(np.nan_to_num(
            data.get('taker_vol_lsr', 0.0) * pd.Series(data.get('vol', 0.0)).rolling(window=24, min_periods=12).apply(
                lambda x: (x - x.mean()) / (x.std() + 1e-12)
            ).fillna(0.0).values
        )),

        # =====================================================================
        # Regime / 市场状态相关因子
        # =====================================================================

        # 长周期方向 regime：基于 96 根收益的符号（约 4 天），-1/0/1
        'regime_trend_96': lambda data: norm(np.nan_to_num(
            np.sign(pd.Series(data['c']).pct_change(96).values)
        )),

        # 波动率 regime：过去 24 根的 realized vol 水平
        'regime_vol_24': lambda data: norm(np.nan_to_num(
            pd.Series(data['c']).pct_change().rolling(window=24, min_periods=12).std().values
        )),

        # 流动性 regime：过去 168 根（约一周）的成交量均值
        'regime_liquidity_168': lambda data: norm(np.nan_to_num(
            pd.Series(data.get('vol', 0.0)).rolling(window=168, min_periods=50).mean().values
        )),
    }

    # 如果指定了类别，则按目录过滤需要返回的特征集合
    if include_categories:
        catalog = get_feature_catalog()
        selected_names = set()
        for cat in include_categories:
            selected_names.update(catalog.get(cat, []))
            
            if cat == 'liquidation':
                features.update(get_advanced_liquidation_features(list(init_ohlcva_df.keys())))
        # 若用户传入不存在的类别，不报错，仅返回空/交集
        features = {k: v for k, v in features.items() if k in selected_names}
    
    return features

    # return {
    #     'lgp_shortcut_illiq_6': lambda data: norm(np.nan_to_num(pd.Series(2*(data['h'] - data['l']) - np.abs(data['c'] - data['o'])).rolling(6, min_periods=1).apply(lambda x: x.mean()))),
    #     'h_ts_std_10': lambda data: norm(np.nan_to_num(pd.Series(data['h']).rolling(window=10, min_periods=5).std())),
    #     'v_ta_cmo_25': lambda data: norm(talib.CMO(data['vol'], 25)),
    #     'v_ts_argrange_10': lambda data: norm(np.nan_to_num(pd.Series(data['vol']).rolling(window=10, min_periods=5).apply(np.argmax) + 1) - np.nan_to_num(pd.Series(data['vol']).rolling(window=10, min_periods=5).apply(np.argmin) + 1)),
    #     'c_ts_argrange_10': lambda data: norm(np.nan_to_num(pd.Series(data['c']).rolling(window=10, min_periods=5).apply(np.argmax) + 1) - np.nan_to_num(pd.Series(data['c']).rolling(window=10, min_periods=5).apply(np.argmin) + 1)),
    #     'c_ts_day_min_40': lambda data: norm(np.nan_to_num(pd.Series(data['c']).rolling(window=41).apply(lambda x: 40 - x.values.tolist()[:-1].index(np.min(x.values.tolist()[:-1]))))),
    #     'l_ts_prod_8': lambda data: norm(np.nan_to_num(pd.Series(data['l']).rolling(window=8, min_periods=4).apply(np.prod))),
    #     'v_ts_std_20': lambda data: norm(np.nan_to_num(pd.Series(data['vol']).rolling(window=20, min_periods=10).std())),
    #     'h_ta_lr_angle_10': lambda data: norm(np.nan_to_num(talib.LINEARREG_ANGLE(data['h'], timeperiod=10))),
    #     'v_ts_sma_21': lambda data: norm(np.nan_to_num(pd.Series(data['vol']).rolling(window=21, min_periods=11).mean())),
    #     'h_ts_kurt_20': lambda data: norm(np.nan_to_num(pd.Series(data['h']).rolling(window=20, min_periods=10).kurt())),
    #     'v_ts_range_5': lambda data: norm(np.nan_to_num(pd.Series(data['vol']).rolling(window=5, min_periods=3).max()) - np.nan_to_num(pd.Series(data['vol']).rolling(window=5, min_periods=3).min())),
    #     'c_ts_sma_21': lambda data: norm(np.nan_to_num(pd.Series(data['c']).rolling(window=21, min_periods=11).mean())),
    #     'o_ts_argmin_10': lambda data: norm(np.nan_to_num(pd.Series(data['o']).rolling(window=10, min_periods=5).apply(np.argmin) + 1)),
    #     'o_ta_lr_slope_10': lambda data: norm(np.nan_to_num(talib.LINEARREG_SLOPE(data['o'], timeperiod=10))),
    #     'l_ts_argmax_20': lambda data: norm(np.nan_to_num(pd.Series(data['l']).rolling(window=20, min_periods=10).apply(np.argmax) + 1)),
    #     'o_ts_range_10': lambda data: norm(np.nan_to_num(pd.Series(data['o']).rolling(window=10, min_periods=5).max()) - np.nan_to_num(pd.Series(data['o']).rolling(window=10, min_periods=5).min())),
    #     'h_ts_argmax_5': lambda data: norm(np.nan_to_num(pd.Series(data['h']).rolling(window=5, min_periods=3).apply(np.argmax) + 1)),
    #     'l_ts_day_min_40': lambda data: norm(np.nan_to_num(pd.Series(data['l']).rolling(window=41).apply(lambda x: 40 - x.values.tolist()[:-1].index(np.min(x.values.tolist()[:-1]))))),
    #     'v_ts_day_max_20': lambda data: norm(np.nan_to_num(pd.Series(data['vol']).rolling(window=21).apply(lambda x: 20 - x.values.tolist()[:-1].index(np.max(x.values.tolist()[:-1]))))),
    #     'c_ta_tsf_5': lambda data: norm(np.nan_to_num(talib.TSF(data['c'], timeperiod=5))),
    #     'c_power_c': lambda data: norm(np.nan_to_num(np.power(data['c'], 3))),
    #     'l_ts_argrange_5': lambda data: norm(np.nan_to_num(pd.Series(data['l']).rolling(window=5, min_periods=3).apply(np.argmax) + 1) - np.nan_to_num(pd.Series(data['l']).rolling(window=5, min_periods=3).apply(np.argmin) + 1)),
    #     'v_power_a': lambda data: norm(np.nan_to_num(np.power(data['vol'], 3))),
    #     'v_cci_25_obv': lambda data: norm(talib.OBV(norm(np.nan_to_num(talib.CCI(data['h'], data['l'], data['c'], timeperiod=25))), data['vol'])),
    #     'v_cci_25_sum': lambda data: norm(np.cumsum(norm(np.nan_to_num(talib.CCI(data['h'], data['l'], data['c'], timeperiod=25))) * data['vol'])),
    #     'ori_trix_8': lambda data: norm(np.nan_to_num(talib.TRIX(data['c'], timeperiod=8))),
    #     'ori_trix_21': lambda data: norm(np.nan_to_num(talib.TRIX(data['c'], timeperiod=21))),
    #     'ori_trix_55': lambda data: norm(np.nan_to_num(talib.TRIX(data['c'], timeperiod=55))),
    #     'v_trix_8_obv': lambda data: norm(talib.OBV(norm(np.nan_to_num(talib.TRIX(data['c'], timeperiod=8))), data['vol'])),
    #     'v_sar_obv': lambda data: norm(talib.OBV(norm(np.nan_to_num(talib.SAR(data['h'], data['l']))), data['vol'])),
    #     'ori_ta_bop': lambda data: norm(talib.BOP(data['o'], data['h'], data['l'], data['c'])),
    #     'v_bop_sum': lambda data: norm(np.cumsum(norm(talib.BOP(data['o'], data['h'], data['l'], data['c'])) * data['vol'])),
    #     'ori_rsi_6': lambda data: norm(talib.RSI(data['c'], timeperiod=6)),
    #     'ori_rsi_12': lambda data: norm(talib.RSI(data['c'], timeperiod=12)),
    #     'ori_rsi_24': lambda data: norm(talib.RSI(data['c'], timeperiod=24)),
    #     'v_rsi_6_obv': lambda data: norm(talib.OBV(norm(talib.RSI(data['c'], timeperiod=6)), data['vol'])),
    #     'v_rsi_6_sum': lambda data: norm(np.cumsum(norm(talib.RSI(data['c'], timeperiod=6)) * data['vol'])),
    #     'v_rsi_12_obv': lambda data: norm(talib.OBV(norm(talib.RSI(data['c'], timeperiod=12)), data['vol'])),
    #     'v_rsi_12_sum': lambda data: norm(np.cumsum(norm(talib.RSI(data['c'], timeperiod=12)) * data['vol'])),
    #     'v_rsi_24_obv': lambda data: norm(talib.OBV(norm(talib.RSI(data['c'], timeperiod=24)), data['vol'])),
    #     'v_rsi_24_sum': lambda data: norm(np.cumsum(norm(talib.RSI(data['c'], timeperiod=24)) * data['vol'])),
    #     'ori_cmo_14': lambda data: norm(np.nan_to_num(talib.CMO(data['c'], timeperiod=14))),
    #     'ori_cmo_25': lambda data: norm(np.nan_to_num(talib.CMO(data['c'], timeperiod=25))),
    #     'v_cmo_14_obv': lambda data: norm(talib.OBV(norm(np.nan_to_num(talib.CMO(data['c'], timeperiod=14))), data['vol'])),
    #     'fm20': lambda data: norm(np.nan_to_num(fm20(data['c'], data['o']))),
    #     'fm30': lambda data: norm(np.nan_to_num(fm30(data['c'], data['o']))),
    #     'fm40': lambda data: norm(np.nan_to_num(fm40(data['c'], data['o']))),
    #     'fm60': lambda data: norm(np.nan_to_num(fm60(data['c'], data['o']))),
    #     'ori_ta_macd': lambda data: norm(norm(np.nan_to_num(talib.MACD(data['c'], fastperiod=12, slowperiod=26, signalperiod=9)[2]))),
    #     'ori_ta_obv': lambda data: norm(np.nan_to_num(talib.OBV(data['c'], data['vol']))),
    #     'ori_ta_ad': lambda data: norm(np.nan_to_num(talib.AD(data['h'], data['l'], data['c'], data['vol']))),
    #     'ori_ta_bop': lambda data: norm(talib.BOP(data['o'], data['h'], data['l'], data['c'])),
    #     'ma8_c': lambda data: norm(np.nan_to_num(talib.MA(data['c'], timeperiod=8, matype=0) / data['c'])),
    #     'ma15_c': lambda data: norm(np.nan_to_num(talib.MA(data['c'], timeperiod=15, matype=0) / data['c'])),
    #     'ma25_c': lambda data: norm(np.nan_to_num(talib.MA(data['c'], timeperiod=25, matype=0) / data['c'])),
    #     'ma35_c': lambda data: norm(np.nan_to_num(talib.MA(data['c'], timeperiod=35, matype=0) / data['c'])),
    #     'ma55_c': lambda data: norm(np.nan_to_num(talib.MA(data['c'], timeperiod=55, matype=0) / data['c'])),
    #     'h_line': lambda data: norm(np.nan_to_num(talib.BBANDS(data['c'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0])),
    #     'm_line': lambda data: norm(np.nan_to_num(talib.BBANDS(data['c'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1])),
    #     'l_line': lambda data: norm(np.nan_to_num(talib.BBANDS(data['c'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2])),
    #     'stdevrate': lambda data: norm((talib.BBANDS(data['c'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0] -
    #                                     talib.BBANDS(data['c'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2]) /
    #                                    (data['c'] * 4)),
    #     'sar_index': lambda data: norm(np.nan_to_num(talib.SAR(data['h'], data['l']))),
    #     'sar_close': lambda data: norm((np.nan_to_num(talib.SAR(data['h'], data['l'])) - data['c']) / data['c']),
    #     'mfi_index': lambda data: norm(np.nan_to_num(talib.MFI(data['h'], data['l'], data['c'], data['vol']))),
    #     'mfi_30': lambda data: norm(np.nan_to_num(talib.MFI(data['h'], data['l'], data['c'], data['vol'], timeperiod=30))),
    #     'ppo': lambda data: norm(np.nan_to_num(talib.PPO(data['c'], fastperiod=12, slowperiod=26, matype=0))),
    #     'ad_index': lambda data: norm(np.nan_to_num(talib.AD(data['h'], data['l'], data['c'], data['vol']))),
    #     'ad_real': lambda data: norm(np.nan_to_num(talib.ADOSC(data['h'], data['l'], data['c'], data['vol'], fastperiod=3, slowperiod=10))),
    #     'tr_index': lambda data: norm(np.nan_to_num(talib.TRANGE(data['h'], data['l'], data['c']))),
    #     'sarext': lambda data: norm(np.nan_to_num(talib.SAREXT(data['h'], data['l'], startvalue=0, offsetonreverse=0,
    #                                                            accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0,
    #                                                            accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0))),
    #     'kdj_d': lambda data: talib.STOCH(data['h'], data['l'], data['c'], 9, 3, 3)[1],
    #     'kdj_k': lambda data: talib.STOCH(data['h'], data['l'], data['c'], 9, 3, 3)[0],
    #     'obv_v': lambda data: norm(np.nan_to_num(talib.OBV(data['c'], data['vol']))),
    #     'volume_macd': lambda data: norm(np.nan_to_num(talib.MACD(data['vol'], fastperiod=12, slowperiod=26, signalperiod=9)[2])),
    #     'close_macd': lambda data: norm(np.nan_to_num(talib.MACD(data['c'], fastperiod=12, slowperiod=26, signalperiod=9)[2])),
    #     'cci_55': lambda data: norm(talib.CCI(data['c'], data['l'], data['h'], 55)),
    #     'cci_25': lambda data: norm(talib.CCI(data['c'], data['l'], data['h'], 25)),
    #     'cci_14': lambda data: norm(talib.CCI(data['c'], data['l'], data['h'], 14)),
    # }
    
    # return {
    #     'lgp_shortcut_illiq_6': lambda data: norm(np.nan_to_num(pd.Series(2*(data['h'] - data['l']) - np.abs(data['c'] - data['o'])).rolling(6, min_periods=1).apply(lambda x: x.mean()))),
    #     'h_ts_std_10': lambda data: norm(np.nan_to_num(pd.Series(data['h']).rolling(window=10, min_periods=5).std())),
    #     'v_ta_cmo_25': lambda data: norm(talib.CMO(data['vol'], 25)),
    #     'v_ts_argrange_10': lambda data: norm(np.nan_to_num(pd.Series(data['vol']).rolling(window=10, min_periods=5).apply(np.argmax) + 1) - np.nan_to_num(pd.Series(data['vol']).rolling(window=10, min_periods=5).apply(np.argmin) + 1)),
    #     'c_ts_argrange_10': lambda data: norm(np.nan_to_num(pd.Series(data['c']).rolling(window=10, min_periods=5).apply(np.argmax) + 1) - np.nan_to_num(pd.Series(data['c']).rolling(window=10, min_periods=5).apply(np.argmin) + 1)),
    #     'c_ts_day_min_40': lambda data: norm(np.nan_to_num(pd.Series(data['c']).rolling(window=41).apply(lambda x: 40 - x.values.tolist()[:-1].index(np.min(x.values.tolist()[:-1]))))),
    #     'l_ts_prod_8': lambda data: norm(np.nan_to_num(pd.Series(data['l']).rolling(window=8, min_periods=4).apply(np.prod))),
    #     'v_ts_std_20': lambda data: norm(np.nan_to_num(pd.Series(data['vol']).rolling(window=20, min_periods=10).std())),
    #     'h_ta_lr_angle_10': lambda data: norm(np.nan_to_num(talib.LINEARREG_ANGLE(data['h'], timeperiod=10))),
    #     'v_ts_sma_21': lambda data: norm(np.nan_to_num(pd.Series(data['vol']).rolling(window=21, min_periods=11).mean())),
    #     'h_ts_kurt_20': lambda data: norm(np.nan_to_num(pd.Series(data['h']).rolling(window=20, min_periods=10).kurt())),
    #     'v_ts_range_5': lambda data: norm(np.nan_to_num(pd.Series(data['vol']).rolling(window=5, min_periods=3).max()) - np.nan_to_num(pd.Series(data['vol']).rolling(window=5, min_periods=3).min())),
    #     'c_ts_sma_21': lambda data: norm(np.nan_to_num(pd.Series(data['c']).rolling(window=21, min_periods=11).mean())),
    #     'o_ts_argmin_10': lambda data: norm(np.nan_to_num(pd.Series(data['o']).rolling(window=10, min_periods=5).apply(np.argmin) + 1)),
    #     'o_ta_lr_slope_10': lambda data: norm(np.nan_to_num(talib.LINEARREG_SLOPE(data['o'], timeperiod=10))),
    #     'l_ts_argmax_20': lambda data: norm(np.nan_to_num(pd.Series(data['l']).rolling(window=20, min_periods=10).apply(np.argmax) + 1)),
    #     'o_ts_range_10': lambda data: norm(np.nan_to_num(pd.Series(data['o']).rolling(window=10, min_periods=5).max()) - np.nan_to_num(pd.Series(data['o']).rolling(window=10, min_periods=5).min())),
    #     'h_ts_argmax_5': lambda data: norm(np.nan_to_num(pd.Series(data['h']).rolling(window=5, min_periods=3).apply(np.argmax) + 1)),
    #     'l_ts_day_min_40': lambda data: norm(np.nan_to_num(pd.Series(data['l']).rolling(window=41).apply(lambda x: 40 - x.values.tolist()[:-1].index(np.min(x.values.tolist()[:-1]))))),
    #     'v_ts_day_max_20': lambda data: norm(np.nan_to_num(pd.Series(data['vol']).rolling(window=21).apply(lambda x: 20 - x.values.tolist()[:-1].index(np.max(x.values.tolist()[:-1]))))),
    #     'c_ta_tsf_5': lambda data: norm(np.nan_to_num(talib.TSF(data['c'], timeperiod=5))),
    #     'c_power_c': lambda data: norm(np.nan_to_num(np.power(data['c'], 3))),
    #     'l_ts_argrange_5': lambda data: norm(np.nan_to_num(pd.Series(data['l']).rolling(window=5, min_periods=3).apply(np.argmax) + 1) - np.nan_to_num(pd.Series(data['l']).rolling(window=5, min_periods=3).apply(np.argmin) + 1)),
    #     'v_power_a': lambda data: norm(np.nan_to_num(np.power(data['vol'], 3))),
    #     'v_cci_25_obv': lambda data: norm(talib.OBV(norm(np.nan_to_num(talib.CCI(data['h'], data['l'], data['c'], timeperiod=25))), data['vol'])),
    #     'v_cci_25_sum': lambda data: norm(np.cumsum(norm(np.nan_to_num(talib.CCI(data['h'], data['l'], data['c'], timeperiod=25))) * data['vol'])),
    #     'ori_trix_8': lambda data: norm(np.nan_to_num(talib.TRIX(data['c'], timeperiod=8))),
    #     'ori_trix_21': lambda data: norm(np.nan_to_num(talib.TRIX(data['c'], timeperiod=21))),
    #     'ori_trix_55': lambda data: norm(np.nan_to_num(talib.TRIX(data['c'], timeperiod=55))),
    #     'v_trix_8_obv': lambda data: norm(talib.OBV(norm(np.nan_to_num(talib.TRIX(data['c'], timeperiod=8))), data['vol'])),
    #     'v_sar_obv': lambda data: norm(talib.OBV(norm(np.nan_to_num(talib.SAR(data['h'], data['l']))), data['vol'])),
    #     'ori_ta_bop': lambda data: norm(talib.BOP(data['o'], data['h'], data['l'], data['c'])),
    #     'v_bop_sum': lambda data: norm(np.cumsum(norm(talib.BOP(data['o'], data['h'], data['l'], data['c'])) * data['vol'])),
    #     'ori_rsi_6': lambda data: norm(talib.RSI(data['c'], timeperiod=6)),
    #     'ori_rsi_12': lambda data: norm(talib.RSI(data['c'], timeperiod=12)),
    #     'ori_rsi_24': lambda data: norm(talib.RSI(data['c'], timeperiod=24)),
    #     'v_rsi_6_obv': lambda data: norm(talib.OBV(norm(talib.RSI(data['c'], timeperiod=6)), data['vol'])),
    #     'v_rsi_6_sum': lambda data: norm(np.cumsum(norm(talib.RSI(data['c'], timeperiod=6)) * data['vol'])),
    #     'v_rsi_12_obv': lambda data: norm(talib.OBV(norm(talib.RSI(data['c'], timeperiod=12)), data['vol'])),
    #     'v_rsi_12_sum': lambda data: norm(np.cumsum(norm(talib.RSI(data['c'], timeperiod=12)) * data['vol'])),
    #     'v_rsi_24_obv': lambda data: norm(talib.OBV(norm(talib.RSI(data['c'], timeperiod=24)), data['vol'])),
    #     'v_rsi_24_sum': lambda data: norm(np.cumsum(norm(talib.RSI(data['c'], timeperiod=24)) * data['vol'])),
    #     'ori_cmo_14': lambda data: norm(np.nan_to_num(talib.CMO(data['c'], timeperiod=14))),
    #     'ori_cmo_25': lambda data: norm(np.nan_to_num(talib.CMO(data['c'], timeperiod=25))),
    #     'v_cmo_14_obv': lambda data: norm(talib.OBV(norm(np.nan_to_num(talib.CMO(data['c'], timeperiod=14))), data['vol'])),
    #     'fm20': lambda data: norm(np.nan_to_num(fm20(data['c'], data['o']))),
    #     'fm30': lambda data: norm(np.nan_to_num(fm30(data['c'], data['o']))),
    #     'fm40': lambda data: norm(np.nan_to_num(fm40(data['c'], data['o']))),
    #     'fm60': lambda data: norm(np.nan_to_num(fm60(data['c'], data['o']))),
    #     'ori_ta_macd': lambda data: norm(norm(np.nan_to_num(talib.MACD(data['c'], fastperiod=12, slowperiod=26, signalperiod=9)[2]))),
    #     'ori_ta_obv': lambda data: norm(np.nan_to_num(talib.OBV(data['c'], data['vol']))),
    #     'ori_ta_ad': lambda data: norm(np.nan_to_num(talib.AD(data['h'], data['l'], data['c'], data['vol']))),
    #     'ori_ta_bop': lambda data: norm(talib.BOP(data['o'], data['h'], data['l'], data['c'])),
    #     'ma8_c': lambda data: norm(np.nan_to_num(talib.MA(data['c'], timeperiod=8, matype=0) / data['c'])),
    #     'ma15_c': lambda data: norm(np.nan_to_num(talib.MA(data['c'], timeperiod=15, matype=0) / data['c'])),
    #     'ma25_c': lambda data: norm(np.nan_to_num(talib.MA(data['c'], timeperiod=25, matype=0) / data['c'])),
    #     'ma35_c': lambda data: norm(np.nan_to_num(talib.MA(data['c'], timeperiod=35, matype=0) / data['c'])),
    #     'ma55_c': lambda data: norm(np.nan_to_num(talib.MA(data['c'], timeperiod=55, matype=0) / data['c'])),
    #     'h_line': lambda data: norm(np.nan_to_num(talib.BBANDS(data['c'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0])),
    #     'm_line': lambda data: norm(np.nan_to_num(talib.BBANDS(data['c'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1])),
    #     'l_line': lambda data: norm(np.nan_to_num(talib.BBANDS(data['c'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2])),
    #     'stdevrate': lambda data: norm((talib.BBANDS(data['c'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0] -
    #                                     talib.BBANDS(data['c'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2]) /
    #                                    (data['c'] * 4)),
    #     'sar_index': lambda data: norm(np.nan_to_num(talib.SAR(data['h'], data['l']))),
    #     'sar_close': lambda data: norm((np.nan_to_num(talib.SAR(data['h'], data['l'])) - data['c']) / data['c']),
    #     'mfi_index': lambda data: norm(np.nan_to_num(talib.MFI(data['h'], data['l'], data['c'], data['vol']))),
    #     'mfi_30': lambda data: norm(np.nan_to_num(talib.MFI(data['h'], data['l'], data['c'], data['vol'], timeperiod=30))),
    #     'ppo': lambda data: norm(np.nan_to_num(talib.PPO(data['c'], fastperiod=12, slowperiod=26, matype=0))),
    #     'ad_index': lambda data: norm(np.nan_to_num(talib.AD(data['h'], data['l'], data['c'], data['vol']))),
    #     'ad_real': lambda data: norm(np.nan_to_num(talib.ADOSC(data['h'], data['l'], data['c'], data['vol'], fastperiod=3, slowperiod=10))),
    #     'tr_index': lambda data: norm(np.nan_to_num(talib.TRANGE(data['h'], data['l'], data['c']))),
    #     'sarext': lambda data: norm(np.nan_to_num(talib.SAREXT(data['h'], data['l'], startvalue=0, offsetonreverse=0,
    #                                                            accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0,
    #                                                            accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0))),
    #     'kdj_d': lambda data: talib.STOCH(data['h'], data['l'], data['c'], 9, 3, 3)[1],
    #     'kdj_k': lambda data: talib.STOCH(data['h'], data['l'], data['c'], 9, 3, 3)[0],
    #     'obv_v': lambda data: norm(np.nan_to_num(talib.OBV(data['c'], data['vol']))),
    #     'volume_macd': lambda data: norm(np.nan_to_num(talib.MACD(data['vol'], fastperiod=12, slowperiod=26, signalperiod=9)[2])),
    #     'close_macd': lambda data: norm(np.nan_to_num(talib.MACD(data['c'], fastperiod=12, slowperiod=26, signalperiod=9)[2])),
    #     'cci_55': lambda data: norm(talib.CCI(data['c'], data['l'], data['h'], 55)),
    #     'cci_25': lambda data: norm(talib.CCI(data['c'], data['l'], data['h'], 25)),
    #     'cci_14': lambda data: norm(talib.CCI(data['c'], data['l'], data['h'], 14)),
    # }

def calc_liquidation_pressure(df, lookback=24):
    """
    计算清算压力因子：空头爆仓量 / 成交量
    """
    # 1. 基础清算占比 (空头爆仓)
    # 加上 1e-8 防止除以零
    short_liq_ratio = df['liquidation_short'] / (df['volume'] + 1e-8)
    
    # 2. 标准化 (Z-Score)
    # 因为爆仓量通常是长尾分布，取对数或者 rolling z-score 更好
    liq_log = np.log1p(short_liq_ratio)
    liq_zscore = (liq_log - liq_log.rolling(lookback).mean()) / (liq_log.rolling(lookback).std() + 1e-8)
    
    return liq_zscore


def get_feature_catalog() -> dict:
    """
    返回特征类别目录，便于选择性挖掘。
    类别建议：
    - momentum: 趋势/动量
    - volatility: 波动/区间/ATR/波动模型
    - reversal: 反转/摆动
    - volume_price: 价量关系/OBV/量能
    - impact: 冲击/缺口/流动性
    - structure: K线结构/通道位置/布林
    - moving_average: 均线相对位置
    - oscillator: 振荡指标（RSI/CCI/CMO/KDJ等）
    - bands: 布林线相关
    - microcycle: fm20/fm30/fm40/fm60 等周期滤波
    """
    catalog = {
        'liq': [
            'liq_zscore'
        ],
        'momentum': [
            'ori_ta_macd', 'close_macd', 'c_ta_tsf_5', 'h_ta_lr_angle_10', 'o_ta_lr_slope_10',
            'v_trix_8_obv', 'ori_trix_8', 'ori_trix_21', 'ori_trix_55', 'obv_lr_slope_20',
            # 'ret_1', 'ret_4', 'ret_12', 'ret_24', 'ret_48', 'ret_96',
            'trend_slope_24', 
            'trend_slope_72', 
            'trend_slope_168',
            'up_ratio_24',
            'donchian_pos_50', 'donchian_pos_200',
        ],
        'volatility': [
            'h_ts_std_10', 'v_ts_std_20', 'v_ts_range_5', 'tr_index', 'atr_14', 'range_over_atr_14',
            'atr_over_maatr_14_50', 'var_parkinson', 'var_garman_klass', 'var_rogers_satchell',
            'bb_width_over_atr_20',
        ],
        'reversal': [
            'ori_rsi_6', 'ori_rsi_12', 'ori_rsi_24', 'kdj_k', 'kdj_d',
        ],
        'volume_price': [
            'ori_ta_obv', 'obv_v', 'volume_macd', 'ret_vol_corr_20', 'v_power_a',
            'v_ts_sma_21', 'v_cci_25_obv', 'v_cci_25_sum', 'v_rsi_6_obv', 'v_rsi_6_sum',
            'v_rsi_12_obv', 'v_rsi_12_sum', 'v_rsi_24_obv', 'v_rsi_24_sum', 'mfi_index', 'mfi_30',
            'vol_level', 'vol_ccy_level', 'avg_trade_size', 'turnover_oi_24',
        ],
        'impact': [
            'gap_strength_14', 'gap_signed_14',
            'amihud_illiq_20', 'amihud_illiq_5', 'amihud_illiq_60',
            'ad_index', 'ad_real', 'lgp_shortcut_illiq_6', 'taker_imbalance',
            'taker_imbalance_vol',
        ],
        'crowding': [
            'oi_zscore_24', 'oi_change_24',
            'toptrader_oi_skew', 'toptrader_count_skew', 'toptrader_oi_skew_abs',
        ],
        'regime': [
            'regime_trend_96', 'regime_vol_24', 'regime_liquidity_168',
        ],
        'structure': [
            'clv', 'body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio', 'donchian_pos_20',
            'sar_index', 'sar_close', 'ori_ta_bop', 'v_bop_sum',
        ],
        'moving_average': [
            'ma8_c', 'ma15_c', 'ma25_c', 'ma35_c', 'ma55_c',
        ],
        'oscillator': [
            'cci_14', 'cci_25', 'cci_55', 'ori_cmo_14', 'ori_cmo_25',
        ],
        'bands': [
            'h_line', 'm_line', 'l_line', 'stdevrate',
        ],
        'microcycle': [
            'fm20', 'fm30', 'fm40', 'fm60',
        ],
    }
    return catalog


def calculate_base_fields(data, base_fields, apply_norm=True, rolling_zscore_window=2000):
    for field, formula in tqdm(base_fields.items(), desc="Processing"):
        if apply_norm:
            data[field] = norm1(formula(data), rolling_zscore_window)
        else:
            data[field] = formula(data)
    return data


def _expanding_zscore(x, ddof=1):
    # 这是相当于expanding z-score标准化，但是这里的标准差是用的无偏估计
    x = np.array(x)
    x = np.nan_to_num(x)
    x_cumsum = np.cumsum(x)
    x_squared_cumsum = np.cumsum(x ** 2)
    count = np.arange(1, len(x) + 1)
    x_mean = x_cumsum / count
    x_std = np.sqrt(((x_squared_cumsum - 2 * x_cumsum * x_mean) / count) + x_mean ** 2) * np.sqrt(
        count / (count - ddof))
    x_value = (x - x_mean) / x_std
    # clip的值，需要测算，没有在log_return 基础上乘1000
    # x_value = np.clip(x_value, -6, 6)
    x_value = np.nan_to_num(x_value)

    return x_value


# ----------k_v1 version-----------------
# def norm1(x, rolling_zscore_window):
#     window = rolling_zscore_window

#     arr = np.asarray(x)
#     x = np.sign(arr) * np.log1p(np.abs(arr)) / np.log1p(np.abs(np.mean(arr)))

#     factors_data = pd.DataFrame(x, columns=['factor'])
#     factors_data = factors_data.replace([np.inf, -np.inf, np.nan], 0.0)
#     # factors_mean = factors_data.rolling(window=window, min_periods=1).mean()
#     factors_std = factors_data.rolling(window=window, min_periods=1).std()
#     # factor_value = (factors_data - factors_mean) / factors_std
#     factor_value = (factors_data ) / factors_std
#     factor_value = factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
#     # factor_value = factor_value.clip(-6, 6)
#     return np.nan_to_num(factor_value).flatten()

def norm1(x, rolling_zscore_window):
    window = rolling_zscore_window
    arr = np.asarray(x)
    
    # 1. Log 变换：只做非线性压缩，不涉及全局统计量
    # 处理长尾分布，保留符号
    x_log = np.sign(arr) * np.log1p(np.abs(arr))
    
    factors_data = pd.DataFrame(x_log, columns=['factor'])
    factors_data = factors_data.replace([np.inf, -np.inf, np.nan], 0.0)
    
    # 2. Rolling Z-Score (必须去均值)
    factors_mean = factors_data.rolling(window=window, min_periods=window//10).mean()
    factors_std = factors_data.rolling(window=window, min_periods=window//10).std()
    
    # 加入 epsilon 防止除零
    factor_value = (factors_data - factors_mean) / (factors_std + 1e-8)
    
    # 3. 异常值截断 (Winsorization)
    factor_value = factor_value.clip(-5, 5)
    
    return np.nan_to_num(factor_value).flatten()

def calculate_features_df(input_df, rolling_zscore_window):
    base_fields = define_base_fields()
    data = calculate_base_fields(input_df.copy(
    ), base_fields, apply_norm=True, rolling_zscore_window=rolling_zscore_window)
    data = data.replace([np.nan, -np.inf, np.inf], 0.0)
    return data


def calculate_features_df_tail(input_df, rolling_zscore_window):
    base_fields = define_base_fields()
    data = calculate_base_fields(
        input_df.copy(), base_fields, apply_norm=False)

    last_row = {}
    # 如下这些是最基础的行情数据，不需要进行norm处理
    base_columns = ['o', 'h', 'l', 'c', 'vol_ccy', 'vol','trades',
           'oi', 'oi_ccy', 'toptrader_count_lsr', 'toptrader_oi_lsr', 'count_lsr',
           'taker_vol_lsr']

    for column in data.columns:
        if column in base_columns:
            last_row[column] = data[column].values[-1]
        else:
            column_data = data[column].values
            if len(column_data) >= rolling_zscore_window:
                column_data = column_data[-rolling_zscore_window:]
            else:
                print(
                    f"Warning: {column} has less than {rolling_zscore_window} values.")
            normalized_data = norm1(column_data, rolling_zscore_window)
            last_row[column] = normalized_data[-1]

    last_row_result = pd.Series(last_row, name=input_df.index[-1])
    last_row_result = last_row_result.replace([np.nan, -np.inf, np.inf], 0.0)
    return last_row_result




class BaseFeature:
    def __init__(self, init_ohlcva_df, include_categories: List[str] = None, rolling_zscore_window: int = 2000):
        # 将所有列转换为 double 类型
        self.init_ohlcva_df = init_ohlcva_df.astype(np.float64)

        self.rolling_zscore_window = rolling_zscore_window
        print('feature 定义')
        # self.base_fields = define_base_fields()
        self.base_fields = define_base_fields(rolling_zscore_window = rolling_zscore_window, include_categories=include_categories)
        print('init_feature 计算')
        self.init_feature_df = self._call(init_ohlcva_df)
        print('init_feature 完成')



    def _call(self, data):
        data = calculate_base_fields(
            data, self.base_fields, apply_norm=False, rolling_zscore_window=self.rolling_zscore_window)
        data = data.replace([np.nan, -np.inf, np.inf], 0.0)
        return data






