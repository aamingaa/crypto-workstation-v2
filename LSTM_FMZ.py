import json
import time
import asyncio
import websockets
from datetime import datetime
import numpy as np
from scipy.stats import skew, kurtosis, wasserstein_distance
import lightgbm as lgb
import pickle
import os
import yaml
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization
import torch
import torch.nn as nn
import warnings
from sklearn.preprocessing import StandardScaler
from numba import jit
import pandas as pd
import hmac
import hashlib
import base64
import urllib.parse
import requests


#   配置管理 

class Config:
    def __init__(self, config_file="config.yaml"):
        # 默认配置
        self.defaults = {
            "trading": {
                "pair": "ETH_USDT",
                "train_bars": 1440,
                "predict_horizon": 10,
                "spread_threshold": 0.002,
                "random_state": 42
            },
            "transformer": {
                "enabled": True, 
                "seq_len": 30,
                "d_model": 32,
                "nhead": 4,
                "num_layers": 2,
                "train_epochs": 10,
                "learning_rate": 0.001
            },
            "optimization": {
                "bayesian_enabled": True,
                "bayesian_iterations": 25,
                "cv_splits": 5
            },
            "monitoring": {
                "decay_threshold": 0.08,
                "feature_drift_threshold": 0.30,
                "auto_retrain_enabled": True,
                "retrain_interval_hours": 24,
                "retrain_min_samples": 50
            },
            "system": {
                "model_dir": "models_v4",
                "state_dir": "strategy_state",
            },
            "signal": {
                "confidence_threshold": 0.65,
                "price_confirm_threshold": 0.0003, 
                "signal_history_limit": 10 
            }
        }
        
        # 加载外部配置文件
        self.config = self.defaults.copy()
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    external_config = yaml.safe_load(f)
                    self._merge_config(self.config, external_config)
                Log(f"成功加载配置文件: {config_file}")
            except Exception as e:
                Log(f"加载配置文件失败: {e}，使用默认配置", "#ff0000")
        else:
            Log(f"配置文件 {config_file} 不存在，使用默认配置")
            # 创建默认配置文件
            self.save_config(config_file)
        
        # 设置属性
        self._set_attributes()
    
    def _merge_config(self, base, update):
        """递归合并配置字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _set_attributes(self):
        """将配置设置为类属性"""
        # Trading
        self.TRADING_PAIR = self.config["trading"]["pair"]
        self.TRAIN_BARS = self.config["trading"]["train_bars"]
        self.PREDICT_HORIZON = self.config["trading"]["predict_horizon"]
        self.SPREAD_THRESHOLD = self.config["trading"]["spread_threshold"]
        self.RANDOM_STATE = self.config["trading"]["random_state"]
        
        # Transformer
        self.TRANSFORMER_ENABLED = self.config["transformer"]["enabled"]
        self.TRANSFORMER_SEQ_LEN = self.config["transformer"]["seq_len"]
        self.TRANSFORMER_D_MODEL = self.config["transformer"]["d_model"]
        self.TRANSFORMER_NHEAD = self.config["transformer"]["nhead"]
        self.TRANSFORMER_NUM_LAYERS = self.config["transformer"]["num_layers"]
        self.TRANSFORMER_TRAIN_EPOCHS = self.config["transformer"]["train_epochs"]
        self.TRANSFORMER_LEARNING_RATE = self.config["transformer"]["learning_rate"]
        
        # Optimization
        self.BAYESIAN_OPT_ENABLED = self.config["optimization"]["bayesian_enabled"]
        self.BAYESIAN_OPT_ITERATIONS = self.config["optimization"]["bayesian_iterations"]
        self.CV_SPLITS = self.config["optimization"]["cv_splits"]
        
        # Monitoring
        self.DECAY_THRESHOLD = self.config["monitoring"]["decay_threshold"]
        self.FEATURE_DRIFT_THRESHOLD = self.config["monitoring"]["feature_drift_threshold"]
        self.AUTO_RETRAIN_ENABLED = self.config["monitoring"]["auto_retrain_enabled"]
        self.RETRAIN_INTERVAL_HOURS = self.config["monitoring"]["retrain_interval_hours"]
        self.RETRAIN_MIN_SAMPLES = self.config["monitoring"]["retrain_min_samples"]
        
        # System
        self.MODEL_DIR = self.config["system"]["model_dir"]
        self.STATE_DIR = self.config["system"]["state_dir"]
        
        # Derived attributes
        self.SYMBOL_API = self.TRADING_PAIR.replace("_", "").lower()
        self.WEBSOCKET_URL = f"wss://fstream.binance.com/stream?streams={self.SYMBOL_API}@aggTrade/{self.SYMBOL_API}@depth20@100ms"

        # 在 _set_attributes(self) 方法中添加：
        self.CONFIDENCE_THRESHOLD = self.config["signal"]["confidence_threshold"]
        self.PRICE_CONFIRM_THRESHOLD = self.config["signal"]["price_confirm_threshold"]
        self.SIGNAL_HISTORY_LIMIT = self.config["signal"]["signal_history_limit"]
    
    def save_config(self, config_file):
        """保存配置到文件"""
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            Log(f"配置已保存到: {config_file}")
        except Exception as e:
            Log(f"保存配置文件失败: {e}", "#ff0000")

# 全局配置实例
config = Config()
class DingTalkNotifier:
    def __init__(self, webhook_url, secret_key=None):
        if not webhook_url:
            raise ValueError("Webhook URL 不能为空")
        self.webhook_url = webhook_url
        self.secret_key = secret_key

    def _generate_signed_url(self):
        if not self.secret_key:
            return self.webhook_url
        
        timestamp = str(round(time.time() * 1000))
        secret_enc = self.secret_key.encode('utf-8')
        string_to_sign = '{}\n{}'.format(timestamp, self.secret_key)
        string_to_sign_enc = string_to_sign.encode('utf-8')
        hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        
        return f"{self.webhook_url}&timestamp={timestamp}&sign={sign}"

    def send_message(self, title, text, at_mobiles=None, is_at_all=False):

        headers = {'Content-Type': 'application/json;charset=utf-8'}
        data = {
            "msgtype": "markdown",
            "markdown": {
                "title": title,
                "text": text
            },
            "at": {
                "atMobiles": at_mobiles if at_mobiles else [],
                "isAtAll": is_at_all
            }
        }
        
        signed_url = self._generate_signed_url()
        
        try:
            response = requests.post(signed_url, headers=headers, data=json.dumps(data), timeout=10)
            response.raise_for_status() 
            result = response.json()
            if result.get("errcode") == 0:
                pass
            else:
                Log(f"钉钉消息发送失败: {result.get('errmsg')}", "#ff0000")
        except requests.exceptions.RequestException as e:
            Log(f"发送钉钉消息时网络异常: {e}", "#ff0000")
        except Exception as e:
            Log(f"发送钉钉消息时发生未知错误: {e}", "#ff0000")

# --- 全局钉钉通知器实例 ---
DINGTALK_WEBHOOK_URL = "" # <--- 修改这里
DINGTALK_SECRET_KEY = "" # 如果没有加签 ，则为 None <--- 修改这里

if DINGTALK_WEBHOOK_URL and "YOUR_ACCESS_TOKEN" not in DINGTALK_WEBHOOK_URL:
    notifier = DingTalkNotifier(DINGTALK_WEBHOOK_URL, DINGTALK_SECRET_KEY)
else:
    notifier = None
    Log("钉钉通知未配置，将不会发送消息。", "#ffff00")
#  分层状态管理 
class ModelRegistry:
    current_model_version = None
    lgbm_model = None
    transformer_model = None
    best_params = None
    feature_names = [] 
    label_map = {0: "上涨", 1: "下跌", 2: "盘整"} 
    #label_map = None
    training_feature_dist = {}
    combined_feature_dim = 0
    scaler = None
    transformer_scaler = None
    model_base_accuracy = 0.0
    last_retrain_timestamp = 0
    next_lgbm_model = None
    next_transformer_model = None
    next_scaler = None
    next_model_version = None
    latest_prediction_proba = None
    latest_feature_values = None
    signal_history = [] 

def update_feature_names_with_transformer():
    """更新特征名称列表以包含 Transformer 特征"""
    base_features = [
        "obv_change_rate", "vpt_zscore_20", "cmf_20", "price_to_vwap_ratio", "price_change_1m", "price_change_5m", 
        "price_change_15m", "volatility_10m", "volatility_30m", "volume_1m", "volume_5m", 
        "volume_change_5m", "rsi_14", "hour_of_day", "alpha_5m", "wobi_10s", "spread_10s", 
        "depth_imbalance_5", "trade_imbalance_10s", "macd", "macd_hist", "bollinger_width", 
        "return_rolling_mean_5", "return_rolling_std_5", "rsi_x_volatility_30m", 
        "trend_strength", "price_skewness_30", "price_kurtosis_30", "atr_14"
    ]
    
    if config.TRANSFORMER_ENABLED:
        transformer_features = [f"transformer_feat_{i}" for i in range(config.TRANSFORMER_D_MODEL)]
        ModelRegistry.feature_names = base_features + transformer_features
    else:
        ModelRegistry.feature_names = base_features
    
    Log(f"特征名称已更新: 共 {len(ModelRegistry.feature_names)} 个特征")

class FeatureStore:
    klines_1min, ticks, order_books = [], [], []
    kline_features_cache = {}

class RealtimeMonitor:
    active_signal = {"active": False, "start_ts": 0, "prediction": -1, "entry_price": 0.0}
    performance_log = {"predictions": [], "real_outcomes": [], "backtest_accuracy": 0.0}
    retrain_needed = False
    data_quality_issues = {"missing_data": 0, "invalid_format": 0, "out_of_range": 0}

class DataPipeline:
    raw_data_queue = asyncio.Queue()
    last_kline_ts = 0
    initial_klines_ready = asyncio.Event()

class KlineMonitor:
    total_generated = 0  # 总共生成的K线数量
    current_count = 0    # 当前存储的K线数量
    last_generated_time = 0  # 最后生成K线的时间
    generation_success = 0   # 成功生成次数
    generation_skipped = 0   # 跳过生成次数（无数据）

#   状态持久化 
class StatePersistence:
    @staticmethod
    def save_state():
        try:
            os.makedirs(config.STATE_DIR, exist_ok=True)
            performance_log = RealtimeMonitor.performance_log
            if performance_log is None:
                performance_log = {"predictions": [], "real_outcomes": [], "backtest_accuracy": 0.0, "live_accuracy": 0.0}
            
            active_signal = RealtimeMonitor.active_signal
            if active_signal is None:
                active_signal = {"active": False, "start_ts": 0, "prediction": -1, "entry_price": 0.0}
            
            data_quality_issues = RealtimeMonitor.data_quality_issues
            if data_quality_issues is None:
                data_quality_issues = {"missing_data": 0, "invalid_format": 0, "out_of_range": 0}
            
            signal_history = ModelRegistry.signal_history
            if signal_history is None:
                signal_history = []
            
            state_data = {
                "timestamp": time.time(),
                "klines_1min": FeatureStore.klines_1min[-1000:],
                "performance_log": performance_log,
                "active_signal": active_signal,
                "last_retrain_timestamp": ModelRegistry.last_retrain_timestamp,
                "current_model_version": ModelRegistry.current_model_version,
                "data_quality_issues": data_quality_issues,
                "signal_history": signal_history
            }
            state_file = os.path.join(config.STATE_DIR, "strategy_state.pkl")
            with open(state_file, 'wb') as f:
                pickle.dump(state_data, f)
            Log(f"策略状态已保存到: {state_file}")
            return True
        except Exception as e:
            Log(f"保存策略状态失败: {e}", "#ff0000")
            return False
    
    @staticmethod
    def load_state():
        try:
            state_file = os.path.join(config.STATE_DIR, "strategy_state.pkl")
            if not os.path.exists(state_file):
                Log("策略状态文件不存在，使用默认状态")
                RealtimeMonitor.performance_log = {"predictions": [], "real_outcomes": [], "backtest_accuracy": 0.0, "live_accuracy": 0.0}
                RealtimeMonitor.active_signal = {"active": False, "start_ts": 0, "prediction": -1, "entry_price": 0.0}
                RealtimeMonitor.data_quality_issues = {"missing_data": 0, "invalid_format": 0, "out_of_range": 0}
                ModelRegistry.signal_history = []
                return False
            
            with open(state_file, 'rb') as f:
                state_data = pickle.load(f)

            if state_data is None:
                Log("状态文件为空，使用默认状态", "#ffff00")
                RealtimeMonitor.performance_log = {"predictions": [], "real_outcomes": [], "backtest_accuracy": 0.0, "live_accuracy": 0.0}
                RealtimeMonitor.active_signal = {"active": False, "start_ts": 0, "prediction": -1, "entry_price": 0.0}
                RealtimeMonitor.data_quality_issues = {"missing_data": 0, "invalid_format": 0, "out_of_range": 0}
                ModelRegistry.signal_history = []
                return False

            FeatureStore.klines_1min = state_data.get("klines_1min", [])
            
            performance_log = state_data.get("performance_log")
            if performance_log is None or not isinstance(performance_log, dict):
                RealtimeMonitor.performance_log = {"predictions": [], "real_outcomes": [], "backtest_accuracy": 0.0, "live_accuracy": 0.0}
            else:
                RealtimeMonitor.performance_log = performance_log
            
            active_signal = state_data.get("active_signal")
            if active_signal is None or not isinstance(active_signal, dict):
                RealtimeMonitor.active_signal = {"active": False, "start_ts": 0, "prediction": -1, "entry_price": 0.0}
            else:
                RealtimeMonitor.active_signal = active_signal
            
            ModelRegistry.last_retrain_timestamp = state_data.get("last_retrain_timestamp", 0)
            ModelRegistry.current_model_version = state_data.get("current_model_version", None)
            
            data_quality_issues = state_data.get("data_quality_issues")
            if data_quality_issues is None or not isinstance(data_quality_issues, dict):
                RealtimeMonitor.data_quality_issues = {"missing_data": 0, "invalid_format": 0, "out_of_range": 0}
            else:
                RealtimeMonitor.data_quality_issues = data_quality_issues
            
            signal_history = state_data.get("signal_history")
            if signal_history is None or not isinstance(signal_history, list):
                ModelRegistry.signal_history = []
            else:
                ModelRegistry.signal_history = signal_history
            
            Log(f"策略状态已从 {state_file} 恢复", "#00ff00")
            return True
        except Exception as e:
            Log(f"加载策略状态失败: {e}", "#ff0000")

            RealtimeMonitor.performance_log = {"predictions": [], "real_outcomes": [], "backtest_accuracy": 0.0, "live_accuracy": 0.0}
            RealtimeMonitor.active_signal = {"active": False, "start_ts": 0, "prediction": -1, "entry_price": 0.0}
            RealtimeMonitor.data_quality_issues = {"missing_data": 0, "invalid_format": 0, "out_of_range": 0}
            ModelRegistry.signal_history = []
            return False

# 优化的特征工程 

BASE_FEATURE_NAMES = [
    "obv_change_rate", "vpt_zscore_20", "cmf_20", "price_to_vwap_ratio", "price_change_1m", "price_change_5m", 
    "price_change_15m", "volatility_10m", "volatility_30m", "volume_1m", "volume_5m", 
    "volume_change_5m", "rsi_14", "hour_of_day", "alpha_5m", "wobi_10s", "spread_10s", 
    "depth_imbalance_5", "trade_imbalance_10s", "macd", "macd_hist", "bollinger_width", 
    "return_rolling_mean_5", "return_rolling_std_5", "rsi_x_volatility_30m", 
    "trend_strength", "price_skewness_30", "price_kurtosis_30", "atr_14"
]

@jit(nopython=True)
def _calculate_ewma(data, span):
    if len(data) == 0:
        return np.array([0.0])
    alpha = 2.0 / (span + 1.0)
    ewma = np.empty_like(data, dtype=np.float64)
    ewma[0] = data[0]
    for i in range(1, len(data)):
        ewma[i] = alpha * data[i] + (1.0 - alpha) * ewma[i-1]
    return ewma

@jit(nopython=True)
def _calculate_macd(closes):
    if len(closes) < 26:
        return 0.0, 0.0
    ema12 = _calculate_ewma(closes, 12)
    ema26 = _calculate_ewma(closes, 26)
    macd_line = ema12 - ema26
    if len(macd_line) < 9:
        return macd_line[-1], 0.0
    signal_line = _calculate_ewma(macd_line, 9)
    macd_hist = macd_line[-1] - signal_line[-1]
    return macd_line[-1], macd_hist

@jit(nopython=True)
def _calculate_rsi(price_changes):
    if len(price_changes) < 14:
        return 50.0
    gains = np.where(price_changes > 0, price_changes, 0)
    losses = np.where(price_changes < 0, -price_changes, 0)
    avg_gain = np.mean(gains[-14:])
    avg_loss = np.mean(losses[-14:])
    if avg_loss == 0: return 100.0
    elif avg_gain == 0: return 0.0
    else:
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

@jit(nopython=True)
def _calculate_atr(highs, lows, closes):
    if len(highs) < 14 or len(lows) < 14 or len(closes) < 14:
        return 0.0
    true_ranges = np.zeros(len(highs) - 1)
    for i in range(1, len(highs)):
        high_low = highs[i] - lows[i]
        high_close = np.abs(highs[i] - closes[i-1])
        low_close = np.abs(lows[i] - closes[i-1])
        true_ranges[i-1] = max(high_low, max(high_close, low_close))
    if len(true_ranges) < 14:
        return 0.0
    return np.mean(true_ranges[-14:])

def calculate_tabular_features_and_labels_vectorized(klines, ticks, order_books, is_realtime=False):
    features, labels = [], []
    required_bars = max(30, config.TRANSFORMER_SEQ_LEN)

    if not is_realtime:
        total_bars_needed = required_bars + config.PREDICT_HORIZON
        if len(klines) < total_bars_needed:
            return None, f"训练K线数据不足，需要 {total_bars_needed} 条（{required_bars} 特征 + {config.PREDICT_HORIZON} 标签），当前 {len(klines)} 条。"
        start_index = required_bars - 1
        end_index = len(klines) - config.PREDICT_HORIZON - 1
        indices_to_process = range(start_index, end_index + 1)
    else:
        if len(klines) < required_bars:
            return None, f"实时K线数据不足，需要 {required_bars} 条，当前 {len(klines)} 条。"
        indices_to_process = [len(klines) - 1]

    klines_array = np.array([[k["open"], k["high"], k["low"], k["close"], k["volume"], k["ts"]] for k in klines])
    
    for i in indices_to_process:
        start_idx = max(0, i - required_bars + 1)
        window_data = klines_array[start_idx:i+1]
        
        if len(window_data) < required_bars:
            continue
            
        opens, highs, lows, closes, volumes, timestamps = window_data.T
        feature_dict = {}
        
        price_changes = np.diff(closes) / (closes[:-1] + 1e-10) if len(closes) > 1 else np.array([0.0])
        feature_dict["price_change_1m"] = price_changes[-1] if len(price_changes) >= 1 else 0.0
        feature_dict["price_change_5m"] = (closes[-1] - closes[-6]) / (closes[-6] + 1e-10) if len(closes) >= 6 else 0.0
        feature_dict["price_change_15m"] = (closes[-1] - closes[-16]) / (closes[-16] + 1e-10) if len(closes) >= 16 else 0.0
        
        feature_dict["volatility_10m"] = np.std(closes[-11:-1]) if len(closes) >= 12 else 0.0  # t-11 到 t-1
        feature_dict["volatility_30m"] = np.std(closes[-31:-1]) if len(closes) >= 32 else 0.0  # t-31 到 t-1
        
        feature_dict["volume_1m"] = volumes[-1]
        feature_dict["volume_5m"] = np.sum(volumes[-5:])
        feature_dict["volume_change_5m"] = (np.sum(volumes[-5:]) - np.sum(volumes[-10:-5])) / (np.sum(volumes[-10:-5]) + 1e-10) if len(volumes) >= 10 else 0.0
        
        if len(closes) > 1:
            obv_changes = np.where(price_changes > 0, volumes[1:], np.where(price_changes < 0, -volumes[1:], 0))
            obv_series = np.cumsum(np.concatenate([[0], obv_changes]))
            if len(obv_series) >= 3:
                obv_change = (obv_series[-2] - obv_series[-3]) / (obv_series[-3] + 1e-10)
            else:
                obv_change = 0.0
        else:
            obv_change = 0.0
        feature_dict["obv_change_rate"] = obv_change
        
        if len(closes) > 1:
            vpt_changes = volumes[1:] * price_changes
            vpt_series = np.cumsum(np.concatenate([[0], vpt_changes]))
            if len(vpt_series) >= 22:  # 需要至少22个值
                vpt_mean = np.mean(vpt_series[-22:-2])  # t-21 到 t-2
                vpt_std = np.std(vpt_series[-22:-2])
                feature_dict["vpt_zscore_20"] = (vpt_series[-2] - vpt_mean) / (vpt_std + 1e-10)
            else:
                feature_dict["vpt_zscore_20"] = 0.0
        else:
            feature_dict["vpt_zscore_20"] = 0.0
        
        if len(closes) > 1:
            mfm = ((closes[:-1] - lows[:-1]) - (highs[:-1] - closes[:-1])) / (highs[:-1] - lows[:-1] + 1e-10)
            mfv = mfm * volumes[:-1]
            if len(mfv) >= 20:
                feature_dict["cmf_20"] = np.sum(mfv[-20:]) / (np.sum(volumes[-21:-1]) + 1e-10)
            else:
                feature_dict["cmf_20"] = 0.0
        else:
            feature_dict["cmf_20"] = 0.0
        
        if len(closes) >= 22:  # 需要至少22根K线
            typical_prices = (highs[-22:-2] + lows[-22:-2] + closes[-22:-2]) / 3  # t-21 到 t-2
            vwap = np.sum(typical_prices * volumes[-22:-2]) / (np.sum(volumes[-22:-2]) + 1e-10)
            feature_dict["price_to_vwap_ratio"] = closes[-2] / (vwap + 1e-10) if len(closes) >= 2 else 1.0
        else:
            feature_dict["price_to_vwap_ratio"] = 1.0
        
        if len(price_changes) >= 15 and len(price_changes) > 1:
            historical_price_changes = price_changes[-15:-1] if len(price_changes) >= 16 else price_changes[:-1]
            feature_dict["rsi_14"] = _calculate_rsi(historical_price_changes)
        else:
            feature_dict["rsi_14"] = 50.0
        
        feature_dict["hour_of_day"] = datetime.fromtimestamp(timestamps[-1] / 1000).hour
        
        current_ts = timestamps[-1]
        ticks_in_5m = [t for t in ticks if t["ts"] >= current_ts - 5 * 60 * 1000 and t["ts"] < current_ts]
        buy_vol_5m = sum(t["qty"] for t in ticks_in_5m if t["side"] == "buy")
        sell_vol_5m = sum(t["qty"] for t in ticks_in_5m if t["side"] == "sell")
        feature_dict["alpha_5m"] = (buy_vol_5m - sell_vol_5m) / (buy_vol_5m + sell_vol_5m + 1e-10) if (buy_vol_5m + sell_vol_5m) != 0 else 0.0
        
        books_in_10s = [b for b in order_books if b["ts"] >= current_ts - 10 * 1000 and b["ts"] < current_ts]
        if books_in_10s:
            latest_book = books_in_10s[-1]
            if latest_book.get("bids") and latest_book["bids"] and latest_book.get("asks") and latest_book["asks"]:
                feature_dict["spread_10s"] = float(latest_book["asks"][0][0]) - float(latest_book["bids"][0][0])
                bid_vol = sum(float(p[1]) for p in latest_book["bids"])
                ask_vol = sum(float(p[1]) for p in latest_book["asks"])
                feature_dict["wobi_10s"] = bid_vol / (bid_vol + ask_vol + 1e-10) if (bid_vol + ask_vol) != 0 else 0.0
                bid_depth_5 = sum(float(p[0]) * float(p[1]) for p in latest_book["bids"][:5])
                ask_depth_5 = sum(float(p[0]) * float(p[1]) for p in latest_book["asks"][:5])
                feature_dict["depth_imbalance_5"] = (bid_depth_5 - ask_depth_5) / (bid_depth_5 + ask_depth_5 + 1e-10) if (bid_depth_5 + ask_depth_5) != 0 else 0.0
            else:
                feature_dict["spread_10s"] = feature_dict["wobi_10s"] = feature_dict["depth_imbalance_5"] = 0.0
        else:
            feature_dict["spread_10s"] = feature_dict["wobi_10s"] = feature_dict["depth_imbalance_5"] = 0.0
        
        ticks_in_10s = [t for t in ticks if t["ts"] >= current_ts - 10 * 1000 and t["ts"] < current_ts]
        buy_qty_10s = sum(t["qty"] for t in ticks_in_10s if t["side"] == "buy")
        sell_qty_10s = sum(t["qty"] for t in ticks_in_10s if t["side"] == "sell")
        sum_qty = buy_qty_10s + sell_qty_10s
        feature_dict["trade_imbalance_10s"] = (buy_qty_10s - sell_qty_10s) / sum_qty if sum_qty != 0 else 0.0

        if len(closes) >= 27:  # 需要至少27根K线
            historical_closes = closes[:-1]  # 使用到t-1的数据
            feature_dict["macd"], feature_dict["macd_hist"] = _calculate_macd(historical_closes)
        else:
            feature_dict["macd"], feature_dict["macd_hist"] = 0.0, 0.0
        
        if len(closes) >= 22:  # 需要至少22根K线
            historical_closes = closes[-22:-2]  # t-21 到 t-2
            ma20 = np.mean(historical_closes)
            std20 = np.std(historical_closes)
            feature_dict["bollinger_width"] = (2 * std20) / (ma20 + 1e-10) if ma20 != 0 else 0.0
        else:
            feature_dict["bollinger_width"] = 0.0
        
        if len(closes) >= 16:  # 需要至少16根K线
            historical_highs = highs[:-1] if len(highs) > 15 else highs
            historical_lows = lows[:-1] if len(lows) > 15 else lows
            historical_closes = closes[:-1] if len(closes) > 15 else closes
            feature_dict["atr_14"] = _calculate_atr(historical_highs, historical_lows, historical_closes)
        else:
            feature_dict["atr_14"] = 0.0
        
        if len(price_changes) >= 7:  # 需要至少7个收益率
            historical_returns = price_changes[-7:-1] if len(price_changes) >= 8 else price_changes[:-1]
            feature_dict["return_rolling_mean_5"] = np.mean(historical_returns[-5:]) if len(historical_returns) >= 5 else 0.0
            feature_dict["return_rolling_std_5"] = np.std(historical_returns[-5:]) if len(historical_returns) >= 5 else 0.0
        else:
            feature_dict["return_rolling_mean_5"] = feature_dict["return_rolling_std_5"] = 0.0
        
        if len(closes) >= 12:  # 需要至少12根K线
            recent_prices = closes[-7:-2] if len(closes) >= 9 else closes[:-1]  # t-6 到 t-2
            past_prices = closes[-12:-7] if len(closes) >= 13 else closes[:len(recent_prices)]  # t-11 到 t-7
            if len(recent_prices) == len(past_prices) and len(recent_prices) > 0:
                rising_count = np.sum(recent_prices > past_prices)
                feature_dict["trend_strength"] = rising_count / len(recent_prices)
            else:
                feature_dict["trend_strength"] = 0.0
        else:
            feature_dict["trend_strength"] = 0.0
        
        feature_dict["rsi_x_volatility_30m"] = feature_dict.get("rsi_14", 0) * feature_dict.get("volatility_30m", 0)
        
        if len(closes) >= 32:  # 需要至少32根K线
            historical_closes = closes[-32:-2]  # t-31 到 t-2
            feature_dict["price_skewness_30"] = skew(historical_closes)
            feature_dict["price_kurtosis_30"] = kurtosis(historical_closes)
        else:
            feature_dict["price_skewness_30"] = feature_dict["price_kurtosis_30"] = 0.0

        # 构建特征向量（返回原始值，不标准化）
        features.append([feature_dict.get(name, 0.0) for name in BASE_FEATURE_NAMES])
        
        if not is_realtime:
            if i + config.PREDICT_HORIZON < len(klines):
                future_price = klines[i + config.PREDICT_HORIZON]["close"]
                current_price = klines[i]["close"]
                if future_price > current_price * (1 + config.SPREAD_THRESHOLD):
                    labels.append(0)
                elif future_price < current_price * (1 - config.SPREAD_THRESHOLD):
                    labels.append(1)
                else:
                    labels.append(2)

    if len(features) > 0:
        features_df = pd.DataFrame(features, columns=BASE_FEATURE_NAMES)
        if not is_realtime:
            return features_df, np.array(labels)
        else:
            return features_df, None  # 实时模式返回未标准化的特征
    else:
        return None, "未能生成任何有效的特征向量，请检查数据源或切片逻辑。"

def get_transformer_input(klines, index):
    if index < config.TRANSFORMER_SEQ_LEN - 1:
        Log(f"Transformer输入数据不足，需要 {config.TRANSFORMER_SEQ_LEN} 条，当前 {index + 1} 条。将使用零填充。", "#ffff00")
        return torch.zeros(1, config.TRANSFORMER_SEQ_LEN, 5)
    
    start_idx = index - config.TRANSFORMER_SEQ_LEN + 1
    end_idx = index + 1
    
    if start_idx < 0:
        Log(f"Transformer输入数据不足，起始索引为负数。将使用零填充。", "#ffff00")
        return torch.zeros(1, config.TRANSFORMER_SEQ_LEN, 5)

    seq_data = klines[start_idx:end_idx]
    ohlcv = np.array([[k["open"], k["high"], k["low"], k["close"], k["volume"]] for k in seq_data])
    
    if ModelRegistry.transformer_scaler is None:
        ModelRegistry.transformer_scaler = StandardScaler()
        # 只有在训练时才fit scaler，实时预测时直接transform
        if len(klines) > config.TRANSFORMER_SEQ_LEN: 
            all_ohlcv = np.array([[k["open"], k["high"], k["low"], k["close"], k["volume"]] for k in klines])
            ModelRegistry.transformer_scaler.fit(all_ohlcv)

    normalized_ohlcv = ModelRegistry.transformer_scaler.transform(ohlcv)
    return torch.tensor(normalized_ohlcv, dtype=torch.float32).unsqueeze(0)

def get_transformer_input_with_fixed_scaler(klines, index, fixed_scaler):
    if index < config.TRANSFORMER_SEQ_LEN - 1:
        Log(f"Transformer输入数据不足，需要 {config.TRANSFORMER_SEQ_LEN} 条，当前 {index + 1} 条。将使用零填充。", "#ffff00")
        return torch.zeros(1, config.TRANSFORMER_SEQ_LEN, 5)
    
    start_idx = index - config.TRANSFORMER_SEQ_LEN + 1
    end_idx = index + 1
    
    if start_idx < 0:
        Log(f"Transformer输入数据不足，起始索引为负数。将使用零填充。", "#ffff00")
        return torch.zeros(1, config.TRANSFORMER_SEQ_LEN, 5)

    seq_data = klines[start_idx:end_idx]
    ohlcv = np.array([[k["open"], k["high"], k["low"], k["close"], k["volume"]] for k in seq_data])
    
    normalized_ohlcv = fixed_scaler.transform(ohlcv)
    return torch.tensor(normalized_ohlcv, dtype=torch.float32).unsqueeze(0)
# 改进的Transformer模型 
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=5, d_model=32, nhead=4, num_encoder_layers=2, dim_feedforward=64, dropout=0.1, num_classes=3):
        super(TimeSeriesTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.feature_layer = nn.Linear(d_model, d_model) # 用于提取特征
        self.classifier = nn.Linear(d_model, num_classes) # 用于Transformer自身的预测（如果需要）

    def forward(self, src, return_features=False):
        src = self.input_proj(src)
        memory = self.transformer_encoder(src)
        features = self.feature_layer(memory[:, -1, :])
        
        if return_features:
            return features
        else:
            return self.classifier(features)
#  联合训练模型 (Joint Training Model)
class JointModel(nn.Module):
    def __init__(self, tabular_dim, transformer_dim=32, hidden_dim=64, num_classes=3):
        super(JointModel, self).__init__()
        self.transformer = TimeSeriesTransformer(
            input_dim=5,
            d_model=transformer_dim,
            nhead=config.TRANSFORMER_NHEAD,
            num_encoder_layers=config.TRANSFORMER_NUM_LAYERS,
            num_classes=num_classes # Transformer的输出类别数
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(tabular_dim + transformer_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, tabular_features, sequence_data):
        transformer_features = self.transformer(sequence_data, return_features=True)
        combined_features = torch.cat([tabular_features, transformer_features], dim=1)
        return self.fusion_layer(combined_features)
#  模型训练与管理 
def train_and_tune_model(X, y, klines_for_transformer_training):
    Log("开始训练和调优模型...")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    
    all_possible_classes = np.array([0, 1, 2])
    unique_classes_in_y = np.unique(y)
    missing_classes = np.setdiff1d(all_possible_classes, unique_classes_in_y)
    if len(missing_classes) > 0:
        Log(f"警告: 训练数据y中缺少类别: {missing_classes}。将添加少量虚拟样本。", "#ffff00")
        dummy_features = np.zeros((len(missing_classes), X.shape[1]))
        dummy_labels = missing_classes
        X = np.vstack((X, dummy_features))
        y = np.hstack((y, dummy_labels))
        shuffle_idx = np.random.permutation(len(y))
        X, y = X[shuffle_idx], y[shuffle_idx]

    early_data_cutoff = int(len(X) * 0.7)
    X_early = X[:early_data_cutoff]
    
    # 创建并拟合scaler（用于后续的交叉验证）
    cv_scaler = StandardScaler()
    cv_scaler.fit(X_early)
    X_cv_scaled = cv_scaler.transform(X)  # 用于交叉验证的标准化数据

    # 更新特征名称（确保与当前特征数量匹配）
    update_feature_names_with_transformer()
    
    if config.TRANSFORMER_ENABLED:
        Log("开始联合训练Transformer和分类器...")
        ModelRegistry.transformer_model = TimeSeriesTransformer(
            input_dim=5, 
            d_model=config.TRANSFORMER_D_MODEL, 
            nhead=config.TRANSFORMER_NHEAD, 
            num_encoder_layers=config.TRANSFORMER_NUM_LAYERS,
            num_classes=3
        )
        optimizer_transformer = torch.optim.Adam(ModelRegistry.transformer_model.parameters(), lr=config.TRANSFORMER_LEARNING_RATE)
        criterion_transformer = nn.CrossEntropyLoss()
        early_klines_cutoff = int(len(klines_for_transformer_training) * 0.7)
        early_ohlcv_data = np.array([[k["open"], k["high"], k["low"], k["close"], k["volume"]] 
                                   for k in klines_for_transformer_training[:early_klines_cutoff]])
        
        ModelRegistry.transformer_scaler = StandardScaler()
        ModelRegistry.transformer_scaler.fit(early_ohlcv_data)

        X_transformer_sequences = []
        y_transformer_sequences = []
        min_klines_needed = config.TRANSFORMER_SEQ_LEN + config.PREDICT_HORIZON - 1
        if len(klines_for_transformer_training) < min_klines_needed:
            Log(f"警告: K线数据不足 {min_klines_needed} 条，无法训练Transformer模型。跳过Transformer训练。", "#ffff00")
            config.TRANSFORMER_ENABLED = False
        else:
            for i in range(len(klines_for_transformer_training) - config.PREDICT_HORIZON - config.TRANSFORMER_SEQ_LEN + 1):
                seq_start = i
                seq_end = i + config.TRANSFORMER_SEQ_LEN
                seq_ohlcv = np.array([[k["open"], k["high"], k["low"], k["close"], k["volume"]] 
                                    for k in klines_for_transformer_training[seq_start:seq_end]])
                
                # 使用预先确定的scaler进行转换
                seq_normalized = ModelRegistry.transformer_scaler.transform(seq_ohlcv)
                X_transformer_sequences.append(seq_normalized)
                
                current_kline_for_label = klines_for_transformer_training[seq_end - 1]
                future_kline_for_label = klines_for_transformer_training[seq_end - 1 + config.PREDICT_HORIZON]
                
                current_price_for_label = current_kline_for_label["close"]
                future_price_for_label = future_kline_for_label["close"]

                if future_price_for_label > current_price_for_label * (1 + config.SPREAD_THRESHOLD):
                    y_transformer_sequences.append(0)
                elif future_price_for_label < current_price_for_label * (1 - config.SPREAD_THRESHOLD):
                    y_transformer_sequences.append(1)
                else:
                    y_transformer_sequences.append(2)

            if not X_transformer_sequences:
                Log("没有足够的序列数据用于Transformer训练，跳过Transformer训练。", "#ffff00")
                config.TRANSFORMER_ENABLED = False
            else:
                X_transformer_sequences = torch.tensor(np.array(X_transformer_sequences), dtype=torch.float32)
                y_transformer_sequences = torch.tensor(np.array(y_transformer_sequences), dtype=torch.long)

                Log(f"开始训练Transformer模型，共 {config.TRANSFORMER_TRAIN_EPOCHS} 轮...")
                for epoch in range(config.TRANSFORMER_TRAIN_EPOCHS):
                    ModelRegistry.transformer_model.train()
                    optimizer_transformer.zero_grad()
                    outputs = ModelRegistry.transformer_model(X_transformer_sequences)
                    loss = criterion_transformer(outputs, y_transformer_sequences)
                    loss.backward()
                    optimizer_transformer.step()
                    if (epoch + 1) % 1 == 0:
                        Log(f"Epoch [{epoch+1}/{config.TRANSFORMER_TRAIN_EPOCHS}], Loss: {loss.item():.4f}")
                Log("Transformer模型训练完成。", "#00ff00")
    
    ModelRegistry.scaler = StandardScaler()
    X_scaled_tabular = ModelRegistry.scaler.fit_transform(X)  # 最终模型使用全部数据

    current_feature_count = X_scaled_tabular.shape[1]
    current_feature_names = ModelRegistry.feature_names[:current_feature_count]
    
    if config.TRANSFORMER_ENABLED and ModelRegistry.transformer_model:
        Log("Transformer已启用，开始为LGBM训练数据提取Transformer特征并拼接...")
        
        ModelRegistry.transformer_model.eval()
        transformer_features_for_lgbm = []
        num_tabular_samples = X_scaled_tabular.shape[0]
        
        if len(klines_for_transformer_training) < num_tabular_samples + config.TRANSFORMER_SEQ_LEN - 1:
            Log(f"警告: K线数据不足以生成所有LGBM训练样本的Transformer特征。需要至少 {num_tabular_samples + config.TRANSFORMER_SEQ_LEN - 1} 条，实际只有 {len(klines_for_transformer_training)} 条。", "#ffff00")
            config.TRANSFORMER_ENABLED = False
            Log("已禁用Transformer特征拼接。", "#ffff00")
            X_final = X_scaled_tabular
        else:
            with torch.no_grad():
                for i in range(num_tabular_samples):
                    kline_index = len(klines_for_transformer_training) - num_tabular_samples + i
                    
                    if kline_index < config.TRANSFORMER_SEQ_LEN - 1:
                        Log(f"警告: K线索引 {kline_index} 不足以生成Transformer特征，使用零填充", "#ffff00")
                        transformer_features = np.zeros(config.TRANSFORMER_D_MODEL)
                    else:
                        transformer_input = get_transformer_input(klines_for_transformer_training, kline_index)
                        features = ModelRegistry.transformer_model(transformer_input, return_features=True).squeeze(0).numpy()
                        transformer_features = features
                    
                    transformer_features_for_lgbm.append(transformer_features)

            transformer_features_np = np.array(transformer_features_for_lgbm)
            
            if transformer_features_np.shape[0] != X_scaled_tabular.shape[0]:
                raise ValueError(f"表格特征和Transformer特征的样本数不匹配: "
                                 f"{X_scaled_tabular.shape[0]} != {transformer_features_np.shape[0]}")

            Log(f"拼接特征：表格特征 shape={X_scaled_tabular.shape}, Transformer特征 shape={transformer_features_np.shape}")
            X_final = np.hstack((X_scaled_tabular, transformer_features_np))
            Log(f"拼接后最终特征矩阵 shape={X_final.shape}")

            # 更新特征名称以包含Transformer特征
            transformer_features_names = [f"transformer_feat_{i}" for i in range(config.TRANSFORMER_D_MODEL)]
            ModelRegistry.feature_names = BASE_FEATURE_NAMES + transformer_features_names
            current_feature_count = X_final.shape[1]
            current_feature_names = ModelRegistry.feature_names[:current_feature_count]
    else:
        X_final = X_scaled_tabular
        # 确保特征名称与实际特征数量匹配
        ModelRegistry.feature_names = BASE_FEATURE_NAMES[:current_feature_count]
        current_feature_names = ModelRegistry.feature_names

    # 重新计算特征分布
    ModelRegistry.training_feature_dist = {}
    for i, name in enumerate(current_feature_names):
        if i < X_final.shape[1]:
            ModelRegistry.training_feature_dist[name] = {"mean": np.mean(X_final[:, i]), "std": np.std(X_final[:, i])}

    # --- LightGBM训练与超参数调优 ---
    
    def lgbm_objective(num_leaves, max_depth, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda):
        params = {
            'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
            'boosting_type': 'gbdt', 'n_estimators': 1000, 'learning_rate': 0.05,
            'num_leaves': int(num_leaves), 'max_depth': int(max_depth),
            'min_child_samples': int(min_child_samples), 'subsample': max(min(subsample, 1), 0),
            'colsample_bytree': max(min(colsample_bytree, 1), 0), 'reg_alpha': max(reg_alpha, 0),
            'reg_lambda': max(reg_lambda, 0), 'random_state': config.RANDOM_STATE,
            'n_jobs': -1, 'verbose': -1
        }
        tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
        accuracies = []
        
        for train_idx, val_idx in tscv.split(X_cv_scaled):  # 使用预先标准化的数据
            X_train, X_val = X_cv_scaled[train_idx], X_cv_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            all_possible_classes_fold = np.array(range(params["num_class"]))
            unique_classes_in_y_train = np.unique(y_train)
            missing_classes_fold = np.setdiff1d(all_possible_classes_fold, unique_classes_in_y_train)
            if len(missing_classes_fold) > 0:
                dummy_features_fold = np.zeros((len(missing_classes_fold), X_train.shape[1]))
                dummy_labels_fold = missing_classes_fold
                X_train = np.vstack((X_train, dummy_features_fold))
                y_train = np.hstack((y_train, dummy_labels_fold))

            X_train_df = pd.DataFrame(X_train, columns=BASE_FEATURE_NAMES[:X_train.shape[1]])
            X_val_df = pd.DataFrame(X_val, columns=BASE_FEATURE_NAMES[:X_val.shape[1]])
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train_df, y_train, eval_set=[(X_val_df, y_val)], 
                     callbacks=[lgb.early_stopping(50, verbose=False)])
            preds = model.predict(X_val_df)
            accuracies.append(accuracy_score(y_val, preds))
        return np.mean(accuracies)

    # 贝叶斯优化部分
    if config.BAYESIAN_OPT_ENABLED:
        Log("启用贝叶斯优化...")
        pbounds = {
            'num_leaves': (20, 200), 'max_depth': (5, 50), 'min_child_samples': (20, 500),
            'subsample': (0.5, 1.0), 'colsample_bytree': (0.5, 1.0),
            'reg_alpha': (0.0, 0.1), 'reg_lambda': (0.0, 0.1),
        }
        optimizer = BayesianOptimization(f=lgbm_objective, pbounds=pbounds, random_state=config.RANDOM_STATE, verbose=0)
        optimizer.maximize(init_points=5, n_iter=config.BAYESIAN_OPT_ITERATIONS)
        best_params = optimizer.max["params"]
        Log(f"贝叶斯优化找到的最佳参数: {best_params}")
        
        final_model_params = {
            'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
            'boosting_type': 'gbdt', 'n_estimators': 1000, 'learning_rate': 0.05,
            'num_leaves': int(best_params['num_leaves']), 'max_depth': int(best_params['max_depth']),
            'min_child_samples': int(best_params['min_child_samples']), 'subsample': best_params['subsample'],
            'colsample_bytree': best_params['colsample_bytree'], 'reg_alpha': best_params['reg_alpha'],
            'reg_lambda': best_params['reg_lambda'], 'random_state': config.RANDOM_STATE, 'n_jobs': -1, 'verbose': -1
        }
        final_model = lgb.LGBMClassifier(**final_model_params)
    else:
        Log("跳过贝叶斯优化，使用默认参数。")
        best_params = {
            'num_leaves': 31, 'max_depth': -1, 'min_child_samples': 20,
            'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_alpha': 0.0, 'reg_lambda': 0.0,
        }
        final_model = lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=config.RANDOM_STATE)
    
    # 训练最终模型
    X_df = pd.DataFrame(X_final, columns=current_feature_names)
    
    final_model.fit(X_df, y, eval_set=[(X_df, y)], eval_metric='multi_logloss', 
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                    feature_name=list(X_df.columns))

    # --- 模型最终分析报告 ---
    Log("--- 模型最终分析报告 ---", "#00BFFF")
    backtest_accuracy_benchmark = 0.0
    try:
        from sklearn.model_selection import cross_val_score
        tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
        scores = cross_val_score(final_model, X_df, y, cv=tscv, scoring='accuracy', n_jobs=-1)
        backtest_accuracy_benchmark = np.mean(scores)
        Log("1. 时间序列交叉验证 (Accuracy):")
        Log(f"   - 各折准确率: {[f'{acc:.2%}' for acc in scores]}")
        Log(f"   - 平均准确率: {backtest_accuracy_benchmark:.2%}", "#00ff00")
        Log(f"   - 准确率标准差: {np.std(scores):.4f} (标准差越小，模型越稳定)", "#ffff00")
    except Exception as e:
        Log(f"计算交叉验证得分时出错: {e}", "#ff0000")

    try:
        Log("2. 特征重要性 (Top 15):")
        if len(current_feature_names) == final_model.n_features_:
            feature_importances = sorted(zip(current_feature_names, final_model.feature_importances_), key=lambda x: x[1], reverse=True)
            for name, importance in feature_importances[:15]:
                if importance > 0:
                    Log(f"   - {name}: {importance}")
        else:
            Log(f"特征名称数量 ({len(current_feature_names)}) 与模型特征数 ({final_model.n_features_}) 不匹配。", "#ffff00")
    except Exception as e:
        Log(f"获取特征重要性时出错: {e}", "#ff0000")
    Log("--- 报告结束 ---", "#00BFFF")

    # --- 模型版本控制和保存 ---
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(config.MODEL_DIR, f"lgbm_model_{version}.pkl")
    transformer_path = os.path.join(config.MODEL_DIR, f"transformer_model_{version}.pth")
    scaler_path = os.path.join(config.MODEL_DIR, f"scaler_{version}.pkl")

    model_package = {
        "model_schema_version": "2.0",
        "strategy_code_version": "v20250807_fmz_final",
        "version": version,
        "lgbm_model": final_model,
        "feature_names": current_feature_names,
        "training_feature_dist": ModelRegistry.training_feature_dist,
        "combined_feature_dim": current_feature_count,
        "label_map": {0: "上涨", 1: "下跌", 2: "盘整"},
        "best_params": best_params,
        "backtest_accuracy": backtest_accuracy_benchmark,
        "config_snapshot": config.config
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    Log(f"LGBM模型已保存到: {model_path}")

    if config.TRANSFORMER_ENABLED and ModelRegistry.transformer_model:
        torch.save(ModelRegistry.transformer_model.state_dict(), transformer_path)
        Log(f"Transformer模型已保存到: {transformer_path}")
    
    if ModelRegistry.scaler:
        with open(scaler_path, 'wb') as f:
            pickle.dump(ModelRegistry.scaler, f)
        Log(f"LGBM Scaler已保存到: {scaler_path}")

    if config.TRANSFORMER_ENABLED and ModelRegistry.transformer_scaler:
        transformer_scaler_path = os.path.join(config.MODEL_DIR, f"transformer_scaler_{version}.pkl")
        with open(transformer_scaler_path, 'wb') as f:
            pickle.dump(ModelRegistry.transformer_scaler, f)
        Log(f"Transformer Scaler已保存到: {transformer_scaler_path}")
    # --- 设置为下一个待热切换的模型 ---
    ModelRegistry.next_lgbm_model = final_model
    ModelRegistry.next_transformer_model = ModelRegistry.transformer_model
    ModelRegistry.next_scaler = ModelRegistry.scaler
    ModelRegistry.next_model_version = version
    ModelRegistry.next_model_base_accuracy = backtest_accuracy_benchmark
    ModelRegistry.last_retrain_timestamp = time.time()
    StatePersistence.save_state()
    Log("模型训练和保存完成。", "#00ff00")
def load_latest_model():
    Log("尝试加载最新模型...")
    if not os.path.exists(config.MODEL_DIR):
        Log("模型目录不存在，无法加载模型。", "#ffff00")
        return False
    model_files = sorted([f for f in os.listdir(config.MODEL_DIR) if f.startswith('lgbm_model_') and f.endswith('.pkl')], reverse=True)
    if not model_files:
        Log("未找到任何LGBM模型文件。", "#ffff00")
        return False
    latest_model_path = os.path.join(config.MODEL_DIR, model_files[0])
    try:
        with open(latest_model_path, "rb") as f:
            model_package = pickle.load(f)
        # 模型兼容性检查
        expected_schema_version = "2.0"
        if model_package.get("model_schema_version") != expected_schema_version:
            Log(f"模型结构版本不兼容！期望 '{expected_schema_version}', 实际 '{model_package.get('model_schema_version')}'。", "#ff0000")
            return False

        # 验证模型对象类型
        if not isinstance(model_package.get("lgbm_model"), lgb.LGBMClassifier):
            Log("加载的模型对象不是有效的LightGBM分类器。", "#ff0000")
            return False
        # --- 开始应用模型包中的数据 ---
        ModelRegistry.lgbm_model = model_package["lgbm_model"]
        ModelRegistry.current_model_version = model_package["version"]
        ModelRegistry.feature_names = model_package.get("feature_names", [])
        ModelRegistry.training_feature_dist = model_package.get("training_feature_dist", {})
        ModelRegistry.combined_feature_dim = model_package.get("combined_feature_dim", 0)
        ModelRegistry.label_map = model_package.get("label_map", {})
        ModelRegistry.best_params = model_package.get("best_params", {})
        ModelRegistry.model_base_accuracy = model_package.get("backtest_accuracy", 0.0)
        ModelRegistry.last_retrain_timestamp = os.path.getmtime(latest_model_path)
        scaler_path = os.path.join(config.MODEL_DIR, f"scaler_{ModelRegistry.current_model_version}.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                ModelRegistry.scaler = pickle.load(f)  
            Log(f"成功加载Scaler: {scaler_path}")
        else:
            Log(f"未找到对应的Scaler文件: {scaler_path}", "#ffff00")
            ModelRegistry.scaler = None

        # --- 优化日志输出 ---
        Log(f"成功加载LGBM模型版本: {ModelRegistry.current_model_version} (回测准确率: {ModelRegistry.model_base_accuracy:.2%})", "#00ff00")
        # 加载Transformer模型
        if config.TRANSFORMER_ENABLED:
            transformer_path = os.path.join(config.MODEL_DIR, f"transformer_model_{ModelRegistry.current_model_version}.pth")
            transformer_scaler_path = os.path.join(config.MODEL_DIR, f"transformer_scaler_{ModelRegistry.current_model_version}.pkl")

            if os.path.exists(transformer_path):
                ModelRegistry.transformer_model = TimeSeriesTransformer(
                    input_dim=5, d_model=config.TRANSFORMER_D_MODEL, nhead=config.TRANSFORMER_NHEAD,
                    num_encoder_layers=config.TRANSFORMER_NUM_LAYERS, num_classes=3
                )
                ModelRegistry.transformer_model.load_state_dict(torch.load(transformer_path))
                ModelRegistry.transformer_model.eval()
                Log(f"成功加载Transformer模型版本: {ModelRegistry.current_model_version}", "#00ff00")
            else:
                Log(f"未找到对应的Transformer模型文件: {transformer_path}", "#ffff00")
                ModelRegistry.transformer_model = None # 确保没有加载旧模型
            
            if os.path.exists(transformer_scaler_path):
                with open(transformer_scaler_path, "rb") as f:
                    ModelRegistry.transformer_scaler = pickle.load(f)
                Log(f"成功加载Transformer Scaler: {transformer_scaler_path}", "#00ff00")
            else:
                Log(f"未找到对应的Transformer Scaler文件: {transformer_scaler_path}", "#ffff00")
                ModelRegistry.transformer_scaler = None # 确保没有加载旧的scaler

        return True
    except Exception as e:
        Log(f"加载模型失败: {e}", "#ff0000")
        return False
#  实时监控与再训练触发 
def check_feature_drift(realtime_features_combined):
    if not ModelRegistry.training_feature_dist:
        return 0.0, "未初始化训练集特征分布"

    drifts = []
    # 检查特征数量是否匹配
    if len(realtime_features_combined) != len(ModelRegistry.feature_names):
        msg = f"特征维度不匹配！模型需要 {len(ModelRegistry.feature_names)} 个特征, 实时计算出 {len(realtime_features_combined)} 个。"
        Log(msg, "#ff0000")
        return 999.0, msg 

    for i, name in enumerate(ModelRegistry.feature_names):
        if name in ModelRegistry.training_feature_dist:
            train_mean = ModelRegistry.training_feature_dist[name]["mean"]
            train_std = ModelRegistry.training_feature_dist[name]["std"]
            
            # 计算标准化漂移
            if train_std > 1e-10: # 避免除以零
                drift = abs(realtime_features_combined[i] - train_mean) / train_std
                drifts.append(drift)
            else:
                if abs(realtime_features_combined[i] - train_mean) > 1e-10:
                    drifts.append(1.0) # 如果实时值不为常数，则认为有漂移
                else:
                    drifts.append(0.0)

    avg_drift = (np.mean(drifts) / 20) if drifts else 0.0
    drift_report = f"平均漂移度: {avg_drift:.4f} (阈值: {config.FEATURE_DRIFT_THRESHOLD})"

    if avg_drift > config.FEATURE_DRIFT_THRESHOLD:
        Log(f"🚨 特征漂移警报！{drift_report}", "#ff0000")
        RealtimeMonitor.retrain_needed = True
    
    return avg_drift, drift_report
async def update_performance_monitor(prediction, entry_price, outcome_ts):
    pm = RealtimeMonitor.performance_log
    future_kline = None
    max_wait_time = config.PREDICT_HORIZON * 60 + 10
    start_wait_time = time.time()
    while time.time() - start_wait_time < max_wait_time:
        future_kline = next((k for k in FeatureStore.klines_1min if k["ts"] >= outcome_ts), None)
        if future_kline: 
            break
        await asyncio.sleep(1)
        
    if not future_kline:
        Log(f"等待超时({max_wait_time}s)，无法找到未来K线以验证结果。", "#ffff00")
        return

    # --- 修复性能监控逻辑 ---
    real_outcome_for_signal = -1
    if future_kline["close"] > entry_price:
        real_outcome_for_signal = 0  # 实际为上涨
    elif future_kline["close"] < entry_price:
        real_outcome_for_signal = 1  # 实际为下跌
        
    if real_outcome_for_signal == -1:
        Log(f"性能监控：价格无变化，无法判断涨跌。本次信号不计入准确率统计。", "#ffff00")
        return
        
    is_correct = (prediction == real_outcome_for_signal)
    real_outcome_academic = 2
    if future_kline["close"] > entry_price * (1 + config.SPREAD_THRESHOLD):
        real_outcome_academic = 0
    elif future_kline["close"] < entry_price * (1 - config.SPREAD_THRESHOLD):
        real_outcome_academic = 1

    pm["predictions"].append(prediction)
    pm["real_outcomes"].append(real_outcome_for_signal) 

    correct_predictions = sum(1 for p, r in zip(pm["predictions"], pm["real_outcomes"]) if p == r)
    live_accuracy = 0.0
    if len(pm["predictions"]) > 0:
        live_accuracy = correct_predictions / len(pm["predictions"])
    pm["live_accuracy"] = live_accuracy
    pm["backtest_accuracy"] = live_accuracy  # 保持兼容性
    
    # 更新日志输出
    result_text = "正确" if is_correct else "错误"
    log_msg = (f"性能监控更新：预测 {ModelRegistry.label_map.get(prediction, '未知')}，"
               f"实际 {ModelRegistry.label_map.get(real_outcome_for_signal, '未知')}。"
               f"信号判断: **{result_text}**")
    Log(log_msg)
    Log(f"当前实盘准确率 (信号方向): {live_accuracy:.2%} (样本数: {len(pm['predictions'])})")
    
    base_accuracy = ModelRegistry.model_base_accuracy
    decay_trigger_point = base_accuracy - config.DECAY_THRESHOLD
    if live_accuracy < decay_trigger_point and len(pm["predictions"]) >= config.RETRAIN_MIN_SAMPLES:
        Log(f"⚠️ 模型性能衰退警报！", "#ff0000")
        Log(f"   - 当前实盘准确率 ({live_accuracy:.2%}) 已低于衰退触发点 ({decay_trigger_point:.2%})。", "#ff0000")
        Log(f"   - (模型回测基准: {base_accuracy:.2%}, 衰退容忍度: {config.DECAY_THRESHOLD:.2%}, 样本数: {len(pm['predictions'])})", "#ff0000")
        RealtimeMonitor.retrain_needed = True
    if len(pm["predictions"]) % 10 == 0:
        StatePersistence.save_state()
#  数据收集与处理 
async def websocket_producer(uri, queue):
    reconnect_delay = 5  # 初始重连延迟
    while True:
        try:
            async with websockets.connect(
                uri, 
                ping_interval=20,  # 每20秒发送一次ping
                ping_timeout=20    # 等待20秒的pong响应
            ) as websocket:
                Log(f"成功连接到WebSocket: {uri}")
                reconnect_delay = 5  # 成功连接后重置延迟
                while True:
                    data = await websocket.recv()
                    try:
                        # 快速解析并放入队列，不做任何耗时操作
                        parsed_data = json.loads(data)
                        if not isinstance(parsed_data, dict):
                            raise ValueError("接收到的数据不是有效的JSON字典")
                        await queue.put(parsed_data)
                    except json.JSONDecodeError as e:
                        Log(f"JSON解析错误: {e}, 原始数据: {data[:100]}...", "#ff0000")
                        RealtimeMonitor.data_quality_issues["invalid_format"] += 1
                    except ValueError as e:
                        Log(f"数据格式错误: {e}, 原始数据: {data[:100]}...", "#ff0000")
                        RealtimeMonitor.data_quality_issues["invalid_format"] += 1
        except websockets.exceptions.ConnectionClosed as e:
            Log(f"WebSocket连接关闭: {e}，将在 {reconnect_delay} 秒后重连...", "#ff0000")
            RealtimeMonitor.data_quality_issues["missing_data"] += 1
        except Exception as e:
            Log(f"WebSocket连接发生未知错误: {e}，将在 {reconnect_delay} 秒后重连...", "#ff0000")
            RealtimeMonitor.data_quality_issues["missing_data"] += 1
        
        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(300, reconnect_delay * 2)  

async def data_consumer(queue):

    while True:
        raw_data = await queue.get()
        if 'stream' in raw_data and 'data' in raw_data:
            stream_type = raw_data['stream']
            data = raw_data['data']
            current_ts = time.time() * 1000

            if '@aggTrade' in stream_type:
                try:
                    tick = {
                        "ts": data["T"],
                        "price": float(data["p"]),
                        "qty": float(data["q"]),
                        "side": "buy" if not data["m"] else "sell"
                    }
                    FeatureStore.ticks.append(tick)
                except (KeyError, ValueError) as e:
                    Log(f"AggTrade数据处理错误: {e}, 数据: {data}", "#ff0000")
                    RealtimeMonitor.data_quality_issues["invalid_format"] += 1
            elif '@depth' in stream_type:
                try:
                    order_book = {
                        "ts": int(current_ts),
                        "bids": [[float(price), float(qty)] for price, qty in data.get("b", [])],
                        "asks": [[float(price), float(qty)] for price, qty in data.get("a", [])]
                    }
                    FeatureStore.order_books.append(order_book)
                except (KeyError, ValueError) as e:
                    Log(f"Depth数据处理错误: {e}, 数据: {data}", "#ff0000")
                    RealtimeMonitor.data_quality_issues["invalid_format"] += 1
            
            if len(FeatureStore.ticks) % 1000 == 0:
                one_hour_ago = current_ts - 3600 * 1000
                FeatureStore.ticks = [t for t in FeatureStore.ticks if t["ts"] > one_hour_ago]
                FeatureStore.order_books = [ob for ob in FeatureStore.order_books if ob["ts"] > one_hour_ago]

async def kline_generator():
    Log("K线生成器已启动。")
    while True:
        now = time.time()
        wait_seconds = 60.5 - (now % 60)
        await asyncio.sleep(wait_seconds)
        current_minute_start_ts = int(time.time() // 60) * 60 * 1000
        last_minute_start_ts = current_minute_start_ts - 60 * 1000
        minute_ticks = [t for t in FeatureStore.ticks if last_minute_start_ts <= t["ts"] < current_minute_start_ts]
        if minute_ticks:
            try:
                # 4. 执行K线合成计算
                minute_ticks.sort(key=lambda x: x['ts'])
                opens = minute_ticks[0]["price"]
                closes = minute_ticks[-1]["price"]
                prices = [t["price"] for t in minute_ticks]
                highs = max(prices)
                lows = min(prices)
                volumes = sum(t["qty"] for t in minute_ticks)
                
                new_kline = {"ts": last_minute_start_ts, "open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}
                FeatureStore.klines_1min.append(new_kline)

                KlineMonitor.total_generated += 1
                KlineMonitor.generation_success += 1
                KlineMonitor.last_generated_time = time.time()

                twenty_four_hours_ago = time.time() * 1000 - 168 * 3600 * 1000
                FeatureStore.klines_1min = [k for k in FeatureStore.klines_1min if k["ts"] > twenty_four_hours_ago]
                KlineMonitor.current_count = len(FeatureStore.klines_1min)
                
            except Exception as e:
                Log(f"K线生成时发生错误: {e}", "#ff0000")
                RealtimeMonitor.data_quality_issues["out_of_range"] += 1
                continue # 如果发生错误，跳过本轮后续的信号检查

            if not DataPipeline.initial_klines_ready.is_set():
                if len(FeatureStore.klines_1min) >= config.TRAIN_BARS:
                    Log(f"已收集到 {len(FeatureStore.klines_1min)}/{config.TRAIN_BARS} 条K线，达到训练要求。发送启动信号...", "#00ff00")
                    DataPipeline.initial_klines_ready.set() 
        else:
            # 如果没有tick数据，也打印日志
            log_time_start = datetime.fromtimestamp(last_minute_start_ts / 1000).strftime('%H:%M:%S')
            log_time_end = datetime.fromtimestamp(current_minute_start_ts / 1000).strftime('%H:%M:%S')
            Log(f"时间段 [{log_time_start}, {log_time_end}): 无tick数据，K线生成跳过。", "#ffff00")
            RealtimeMonitor.data_quality_issues["missing_data"] += 1
            # +++ 更新跳过计数 +++
            KlineMonitor.generation_skipped += 1
# 11. 主程序 
async def main_async():
    global config
    config = Config() 
    Log("策略启动...")
    
    # 添加全局初始化 
    def initialize_globals():
        """确保所有全局对象都已初始化"""
        if RealtimeMonitor.performance_log is None:
            RealtimeMonitor.performance_log = {"predictions": [], "real_outcomes": [], "backtest_accuracy": 0.0, "live_accuracy": 0.0}
        if RealtimeMonitor.data_quality_issues is None:
            RealtimeMonitor.data_quality_issues = {"missing_data": 0, "invalid_format": 0, "out_of_range": 0}
        if RealtimeMonitor.active_signal is None:
            RealtimeMonitor.active_signal = {"active": False, "start_ts": 0, "prediction": -1, "entry_price": 0.0}
        if RealtimeMonitor.retrain_needed is None:
            RealtimeMonitor.retrain_needed = False
        if ModelRegistry.signal_history is None:
            ModelRegistry.signal_history = []
        if ModelRegistry.label_map is None:
            ModelRegistry.label_map = {0: "上涨", 1: "下跌", 2: "盘整"}

    if notifier:
        notifier.send_message("策略启动通知", f"## {config.TRADING_PAIR} 策略已启动\n\n> 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    initialize_globals()     
    StatePersistence.load_state()
    
    update_feature_names_with_transformer()
    if not load_latest_model():
        Log("未找到可用模型或模型加载失败，将进入数据收集模式并等待训练。", "#ffff00")
        if len(FeatureStore.klines_1min) < config.TRAIN_BARS:
            Log(f"当前K线 {len(FeatureStore.klines_1min)}/{config.TRAIN_BARS}，等待数据收集中...")

            async def status_updater():
                while not DataPipeline.initial_klines_ready.is_set():
                    status_table = {
                        'type': 'table', 'title': '策略状态', 'cols': ['指标', '值'],
                        'rows': [
                            ['状态', '数据收集中...'],
                            ['K线进度', f'{len(FeatureStore.klines_1min)}/{config.TRAIN_BARS}'],
                            ['目标K线', config.TRAIN_BARS]
                        ]
                    }
                    LogStatus('`' + json.dumps(status_table) + '`')
                    await asyncio.sleep(10)

            updater_task = asyncio.create_task(status_updater())
            await DataPipeline.initial_klines_ready.wait()
            updater_task.cancel()
            Log("数据收集完成，状态更新任务已停止。")
        
        # --- 首次训练 ---
        Log("已收集到足够K线数据，开始特征工程和模型训练。")
        X_train, y_train = calculate_tabular_features_and_labels_vectorized(
            FeatureStore.klines_1min, FeatureStore.ticks, FeatureStore.order_books, is_realtime=False
        )
        
        if X_train is None or y_train is None or len(X_train) == 0:
            Log(f"训练数据生成失败: {y_train if isinstance(y_train, str) else '未知错误'}。无法进行模型训练。", "#ff0000")
            return
        
        ModelRegistry.combined_feature_dim = X_train.shape[1]
        train_and_tune_model(X_train, y_train, FeatureStore.klines_1min)

        if ModelRegistry.next_lgbm_model and ModelRegistry.scaler:
            Log("首次训练完成，直接应用新模型...")
            ModelRegistry.lgbm_model = ModelRegistry.next_lgbm_model
            ModelRegistry.transformer_model = ModelRegistry.next_transformer_model
            ModelRegistry.scaler = ModelRegistry.next_scaler
            ModelRegistry.current_model_version = ModelRegistry.next_model_version
            ModelRegistry.model_base_accuracy = ModelRegistry.next_model_base_accuracy
            
            # 设置特征名称
            if hasattr(ModelRegistry.lgbm_model, 'feature_names_'):
                ModelRegistry.feature_names = ModelRegistry.lgbm_model.feature_names_
            else:
                update_feature_names_with_transformer()
                Log("警告: 新加载的LGBM模型没有 feature_names_ 属性，使用全局更新的特征名称。", "#ffff00")
            
            ModelRegistry.next_lgbm_model, ModelRegistry.next_transformer_model, ModelRegistry.next_scaler, ModelRegistry.next_model_version = None, None, None, None
            Log("新模型已激活。", "#00ff00")
        else:
            Log("模型训练失败或未生成新模型，无法继续。", "#ff0000")
            return

    #  进入实时预测模式的主循环 
    Log("进入实时预测模式。")
    if ModelRegistry.scaler is None:
        Log("错误: Scaler未加载，无法进行特征缩放。策略无法继续运行。", "#ff0000")
        return

    while True:
        # 确保监控对象已初始化
        if RealtimeMonitor.performance_log is None:
            RealtimeMonitor.performance_log = {"predictions": [], "real_outcomes": [], "backtest_accuracy": 0.0, "live_accuracy": 0.0}
        if RealtimeMonitor.data_quality_issues is None:
            RealtimeMonitor.data_quality_issues = {"missing_data": 0, "invalid_format": 0, "out_of_range": 0}
        if RealtimeMonitor.active_signal is None:
            RealtimeMonitor.active_signal = {"active": False, "start_ts": 0, "prediction": -1, "entry_price": 0.0}
            
        now = time.time()
        wait_seconds = 60.5 - (now % 60)
        await asyncio.sleep(wait_seconds)

        # --- 模型热切换 ---
        if ModelRegistry.next_lgbm_model:
            Log(f"执行模型热切换，从 {ModelRegistry.current_model_version} 切换到 {ModelRegistry.next_model_version}")
            if notifier:
                notifier.send_message("模型更新通知", f"## 模型已热切换\n\n- **交易对**: {config.TRADING_PAIR}\n- **旧版本**: {ModelRegistry.current_model_version}\n- **新版本**: {ModelRegistry.next_model_version}\n- **新模型准确率**: {ModelRegistry.next_model_base_accuracy:.2%}")

            ModelRegistry.lgbm_model = ModelRegistry.next_lgbm_model
            ModelRegistry.transformer_model = ModelRegistry.next_transformer_model
            ModelRegistry.scaler = ModelRegistry.next_scaler
            ModelRegistry.current_model_version = ModelRegistry.next_model_version
            ModelRegistry.model_base_accuracy = ModelRegistry.next_model_base_accuracy
            
            # 设置特征名称
            if hasattr(ModelRegistry.lgbm_model, 'feature_names_'):
                ModelRegistry.feature_names = ModelRegistry.lgbm_model.feature_names_
            else:
                update_feature_names_with_transformer()
                Log("警告: 新加载的LGBM模型没有 feature_names_ 属性，使用全局更新的特征名称。", "#ffff00")
            
            ModelRegistry.next_lgbm_model, ModelRegistry.next_transformer_model, ModelRegistry.next_scaler, ModelRegistry.next_model_version = None, None, None, None
            ModelRegistry.next_model_base_accuracy = 0.0 
            Log(f"模型热切换完成。新模型基准准确率: {ModelRegistry.model_base_accuracy:.2%}", "#00ff00")

        # --- 自动再训练检查 ---
        if config.AUTO_RETRAIN_ENABLED and ModelRegistry.lgbm_model:
            time_since_last_retrain = (time.time() - ModelRegistry.last_retrain_timestamp) / 3600
            if RealtimeMonitor.retrain_needed or time_since_last_retrain >= config.RETRAIN_INTERVAL_HOURS:
                Log("触发自动再训练：性能衰退或达到再训练间隔。")
                RealtimeMonitor.performance_log = {"predictions": [], "real_outcomes": [], "backtest_accuracy": 0.0, "live_accuracy": 0.0}
                RealtimeMonitor.retrain_needed = False
                
                update_feature_names_with_transformer()
                X_train, y_train = calculate_tabular_features_and_labels_vectorized(
                    FeatureStore.klines_1min, FeatureStore.ticks, FeatureStore.order_books, is_realtime=False
                )
                if X_train is None or y_train is None or len(X_train) == 0:
                    Log(f"再训练数据生成失败: {y_train if isinstance(y_train, str) else '未知错误'}。跳过本次再训练。", "#ff0000")
                else:
                    Log("开始执行自动再训练...")
                    ModelRegistry.combined_feature_dim = X_train.shape[1]
                    train_and_tune_model(X_train, y_train, FeatureStore.klines_1min)
                    if not ModelRegistry.next_lgbm_model:
                        Log("自动再训练失败，将继续使用旧模型。", "#ff0000")
                    else:
                        Log("自动再训练完成，新模型已准备好进行热切换。", "#00ff00")


        #     实时预测逻辑 


        if not ModelRegistry.lgbm_model:
            Log("模型未加载，无法进行预测。", "#ffff00")
            continue
        X_realtime_tabular_df, error_msg = calculate_tabular_features_and_labels_vectorized(
            FeatureStore.klines_1min, FeatureStore.ticks, FeatureStore.order_books, is_realtime=True
        )

        if X_realtime_tabular_df is None or X_realtime_tabular_df.empty:
            Log(f"实时表格特征计算失败: {error_msg}。跳过本次预测。", "#ffff00")
            continue
        try:
            available_features = [f for f in BASE_FEATURE_NAMES if f in X_realtime_tabular_df.columns]
            if len(available_features) != len(BASE_FEATURE_NAMES):
                Log(f"警告: 实时数据缺少某些特征，期望 {len(BASE_FEATURE_NAMES)}，实际 {len(available_features)}", "#ffff00")
            
            base_features_values = X_realtime_tabular_df[available_features].values
            scaled_base_features = ModelRegistry.scaler.transform(base_features_values)
            scaled_features_df = pd.DataFrame(scaled_base_features, columns=available_features)
        except Exception as e:
            Log(f"特征缩放失败: {e}", "#ff0000")
            continue
        model_expects_transformer = any("transformer_feat" in name for name in ModelRegistry.feature_names)
        transformer_feature_np = None
        if model_expects_transformer:
            if config.TRANSFORMER_ENABLED and ModelRegistry.transformer_model and ModelRegistry.transformer_scaler:
                if len(FeatureStore.klines_1min) >= config.TRANSFORMER_SEQ_LEN:
                    transformer_input = get_transformer_input_with_fixed_scaler(
                        FeatureStore.klines_1min, 
                        len(FeatureStore.klines_1min) - 1,
                        ModelRegistry.transformer_scaler
                    )
                    with torch.no_grad():
                        transformer_feature_np = ModelRegistry.transformer_model(transformer_input, return_features=True).squeeze(0).numpy()
                else:
                    Log(f"K线数据不足 {config.TRANSFORMER_SEQ_LEN} 条，无法生成Transformer特征。", "#ffff00")
                    transformer_feature_np = np.zeros(config.TRANSFORMER_D_MODEL)
            else:
                Log("模型期望Transformer特征，但全局配置禁用或模型文件丢失。使用零填充。", "#ffff00")
                transformer_feature_np = np.zeros(config.TRANSFORMER_D_MODEL)
        try:
            if transformer_feature_np is not None:
                transformer_cols = [f"transformer_feat_{i}" for i in range(config.TRANSFORMER_D_MODEL)]
                transformer_df = pd.DataFrame(transformer_feature_np.reshape(1, -1), columns=transformer_cols)
                final_features_for_model_df = pd.concat([scaled_features_df, transformer_df], axis=1)
                for feature in ModelRegistry.feature_names:
                    if feature not in final_features_for_model_df.columns:
                        final_features_for_model_df[feature] = 0.0
                        
                final_features_for_model_df = final_features_for_model_df[ModelRegistry.feature_names]
            else:
                for feature in ModelRegistry.feature_names:
                    if feature not in scaled_features_df.columns and feature.startswith("transformer_feat_"):
                        scaled_features_df[feature] = 0.0
                final_features_for_model_df = scaled_features_df[ModelRegistry.feature_names]
                
        except Exception as e:
            Log(f"特征拼接失败: {e}", "#ff0000")
            continue
        final_features_np = final_features_for_model_df.values
        avg_drift, drift_report = check_feature_drift(final_features_np[0])
        
        if avg_drift > config.FEATURE_DRIFT_THRESHOLD * 1.5:
            Log(f"漂移过高，暂停本轮预测。{drift_report}", "#ff0000")
            continue
        try:
            if len(final_features_for_model_df.columns) != len(ModelRegistry.feature_names):
                Log(f"严重错误: 最终特征数量({len(final_features_for_model_df.columns)})与模型期望({len(ModelRegistry.feature_names)})不匹配！", "#ff0000")
                continue
                
            predicted_label = ModelRegistry.lgbm_model.predict(final_features_for_model_df)[0]
            predicted_proba = ModelRegistry.lgbm_model.predict_proba(final_features_for_model_df)[0]
            
        except Exception as e:
            Log(f"模型预测失败: {e}", "#ff0000")
            continue

        confidence = np.max(predicted_proba)
        ModelRegistry.latest_prediction_proba = predicted_proba
        ModelRegistry.latest_feature_values = final_features_for_model_df.iloc[0].to_dict()

        if FeatureStore.klines_1min:
            current_kline_ts = FeatureStore.klines_1min[-1]["ts"]
            current_price = FeatureStore.klines_1min[-1]["close"]
        else:
            Log("K线数据为空，无法获取当前价格和时间。","#ff0000")
            continue      
        is_valid = True
        reason = "验证通过"
        if confidence < config.CONFIDENCE_THRESHOLD:
            is_valid = False
            reason = f"置信度不足({confidence:.2f})"
        elif predicted_label != 2:
            if predicted_label == 0:
                if len(FeatureStore.klines_1min) < 2 or current_price <= FeatureStore.klines_1min[-2]["close"] * (1 - config.PRICE_CONFIRM_THRESHOLD):
                    is_valid = False
                    reason = "价格未确认上涨"
            elif predicted_label == 1:
                if len(FeatureStore.klines_1min) < 2 or current_price >= FeatureStore.klines_1min[-2]["close"] * (1 + config.PRICE_CONFIRM_THRESHOLD):
                    is_valid = False
                    reason = "价格未确认下跌"
        signal_record = {
            "timestamp": datetime.fromtimestamp(current_kline_ts / 1000).strftime("%H:%M:%S"),
            "direction": ModelRegistry.label_map.get(predicted_label, '未知'),
            "confidence": f"{confidence:.2%}",
            "valid": "是" if is_valid else "否",
            "reason": reason,
            "price": f"{current_price:.2f}"
        }
        ModelRegistry.signal_history.append(signal_record)
        if len(ModelRegistry.signal_history) > config.SIGNAL_HISTORY_LIMIT:
            ModelRegistry.signal_history.pop(0)
        if int(time.time()) % 10 < 2:
            signal_status_rows = []
            if ModelRegistry.signal_history:
                latest_signal = ModelRegistry.signal_history[-1]
                signal_status_rows = [
                    ["最新信号时间", latest_signal["timestamp"]],
                    ["最新信号方向", latest_signal["direction"]],
                    ["最新信号置信度", latest_signal["confidence"]],
                    ["信号是否有效", latest_signal["valid"]],
                    ["无效原因", latest_signal["reason"]]
                ]
            else:
                signal_status_rows = [["状态", "等待第一个信号中..."]]
            
            signal_status_table = {"type": "table", "title": "交易信号状态", "cols": ["指标", "值"], "rows": signal_status_rows}
            signal_history_rows = []
            for s in reversed(ModelRegistry.signal_history):
                signal_history_rows.append([
                    s["timestamp"],
                    s["direction"],
                    s["confidence"],
                    s["valid"],
                    s["reason"],
                    s["price"]
                ])
            signal_history_table = {"type": "table", "title": f"信号历史 (最近{config.SIGNAL_HISTORY_LIMIT}条)", "cols": ["时间", "方向", "置信度", "有效", "原因", "价格"], "rows": signal_history_rows}
            collection_progress = min(100, (KlineMonitor.current_count / config.TRAIN_BARS) * 100) if config.TRAIN_BARS > 0 else 0
            
            dq = RealtimeMonitor.data_quality_issues
            missing_data = dq.get("missing_data", 0) if dq else 0
            invalid_format = dq.get("invalid_format", 0) if dq else 0
            out_of_range = dq.get("out_of_range", 0) if dq else 0
            
            strategy_status_rows = [
                ["时间", datetime.fromtimestamp(current_kline_ts / 1000).strftime("%Y-%m-%d %H:%M:%S")],
                ["当前价格", f"{current_price:.2f}"],
                ["预测方向", f"{ModelRegistry.label_map.get(predicted_label, '未知')}"],
                ["整体置信度", f"{confidence:.2%}"],
                ["模型版本", ModelRegistry.current_model_version if ModelRegistry.current_model_version else 'N/A'],
                ["实盘准确率", f"{RealtimeMonitor.performance_log.get('live_accuracy', 0.0) if RealtimeMonitor.performance_log else 0.0:.2%}"],
                ['数据质量(缺/错/越)', f'{missing_data}/{invalid_format}/{out_of_range}'],
                ["特征漂移", f"{avg_drift:.4f}"]
            ]
            strategy_status_table = {'type': 'table', 'title': '策略状态', 'cols': ['指标', '值'], 'rows': strategy_status_rows}
            confidence_rows = [[ModelRegistry.label_map.get(i, f'方向{i}'), f'{p:.2%}'] for i, p in enumerate(predicted_proba)]
            confidence_table = {"type": "table", "title": "置信度分布", "cols": ["方向", "概率"], "rows": confidence_rows}
            feature_value_rows = []
            if ModelRegistry.lgbm_model and ModelRegistry.latest_feature_values:
                feature_importances = sorted(zip(ModelRegistry.feature_names, ModelRegistry.lgbm_model.feature_importances_), key=lambda x: x[1], reverse=True)
                for name, _ in feature_importances[:60]: 
                    if name in ModelRegistry.latest_feature_values:
                        value = ModelRegistry.latest_feature_values[name]
                        feature_value_rows.append([name, f'{value:.4f}'])
            feature_value_table = {'type': 'table', 'title': '重要特征值', 'cols': ['特征', '数值'], 'rows': feature_value_rows}
            LogStatus('`' + json.dumps([signal_status_table, signal_history_table, strategy_status_table, confidence_table, feature_value_table]) + '`')
        if predicted_label != 2 and confidence > 0.65 and is_valid and not RealtimeMonitor.active_signal["active"]:
            RealtimeMonitor.active_signal = {"active": True, "start_ts": current_kline_ts, "prediction": predicted_label, "entry_price": current_price}
            Log(f"发出交易信号: {ModelRegistry.label_map.get(predicted_label, '未知')} @ {current_price:.2f} (置信度: {confidence:.2%})", "#0000ff")
            if notifier:
                direction = ModelRegistry.label_map.get(predicted_label, '未知')
                title = f"📈 新交易信号: {direction}" if predicted_label == 0 else f"📉 新交易信号: {direction}"
                live_accuracy = RealtimeMonitor.performance_log.get("live_accuracy", 0.0)
                sample_count = len(RealtimeMonitor.performance_log.get("predictions", []))
                if predicted_label == 0:
                    color = "red"
                elif predicted_label == 1:
                    color = "green"
                else:
                    color = "gray"
                text = (
                    f"## {config.TRADING_PAIR} 交易信号\n\n"
                    f"## <font color='{color}'>方向: {direction}</font>\n\n"
                    f"- **价格**: {current_price:.4f}\n"
                    f"- **置信度**: {confidence:.2%}\n"
                    f"- **模型版本**: {ModelRegistry.current_model_version}\n"
                    f"- **时间**: {datetime.fromtimestamp(current_kline_ts / 1000).strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    f"---\n\n"
                    f"> **当前实盘准确率**: {live_accuracy:.2%} (样本数: {sample_count})"
                )
                loop = asyncio.get_running_loop()
                loop.run_in_executor(None, notifier.send_message, title, text)
            asyncio.create_task(update_performance_monitor(predicted_label, current_price, current_kline_ts + config.PREDICT_HORIZON * 60 * 1000))
        elif predicted_label == 2 and RealtimeMonitor.active_signal["active"]:
            Log("盘整信号，取消当前活跃信号。")
            RealtimeMonitor.active_signal["active"] = False
#  启动程序
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(websocket_producer(config.WEBSOCKET_URL, DataPipeline.raw_data_queue))
    loop.create_task(data_consumer(DataPipeline.raw_data_queue))
    loop.create_task(kline_generator())
    loop.run_until_complete(main_async())