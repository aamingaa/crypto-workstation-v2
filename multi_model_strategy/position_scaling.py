"""
仓位缩放模块
包含 Regime 层、Risk 层、Kelly bet sizing
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class RegimeScaler:
    """
    Regime 层：基于趋势强度和波动水平调整仓位
    """
    
    def __init__(self, config):
        self.config = config
        self.regime_scaler_train = None
        self.regime_scaler_test = None
    
    def build(self, feature_df, train_len):
        """
        构建 Regime 缩放因子
        
        Args:
            feature_df (pd.DataFrame): 特征矩阵（全样本）
            train_len (int): 训练集长度
        
        Returns:
            tuple: (regime_scaler_train, regime_scaler_test)
        """
        print("正在构建 Regime 层缩放因子...")
        n = len(feature_df)
        
        # 1) 趋势因子
        cfg_trend_cols = self.config.get('regime_trend_cols')
        if cfg_trend_cols:
            trend_cols = [c for c in cfg_trend_cols if c in feature_df.columns]
        else:
            trend_cols = [c for c in feature_df.columns if 'regime_trend' in c.lower()]
            if not trend_cols:
                trend_cols = [c for c in feature_df.columns
                              if 'trend_slope_96' in c.lower() or 'trend_slope_72' in c.lower()]
        
        if trend_cols:
            trend_raw = feature_df[trend_cols[0]].values.astype(float)
        else:
            trend_raw = np.zeros(n, dtype=float)
        
        trend_raw = np.nan_to_num(trend_raw, nan=0.0, posinf=0.0, neginf=0.0)
        trend_abs = np.clip(np.abs(trend_raw), 0.0, 3.0)
        trend_score = trend_abs / 3.0
        
        # 2) 波动因子
        cfg_vol_cols = self.config.get('regime_vol_cols')
        if cfg_vol_cols:
            vol_cols = [c for c in cfg_vol_cols if c in feature_df.columns]
        else:
            vol_cols = [c for c in feature_df.columns if 'regime_vol' in c.lower()]
        
        if vol_cols:
            vol_raw = feature_df[vol_cols[0]].values.astype(float)
        else:
            vol_raw = np.zeros(n, dtype=float)
        
        vol_raw = np.nan_to_num(vol_raw, nan=0.0, posinf=0.0, neginf=0.0)
        vol_pos = np.maximum(vol_raw, 0.0)
        a = 0.5
        vol_penalty = 1.0 / (1.0 + a * vol_pos)
        
        # 3) 综合 Regime 缩放
        regime_scaler = trend_score * vol_penalty
        regime_scaler = np.clip(regime_scaler, 0.0, 1.0)
        
        self.regime_scaler_train = regime_scaler[:train_len]
        self.regime_scaler_test = regime_scaler[train_len:]
        
        print(f"Regime 缩放构建完成："
              f"train_mean={self.regime_scaler_train.mean():.3f}, "
              f"test_mean={self.regime_scaler_test.mean():.3f}")
        
        return self.regime_scaler_train, self.regime_scaler_test


class RiskScaler:
    """
    Risk 层：基于拥挤度、冲击、资金成本调整仓位
    """
    
    def __init__(self, config):
        self.config = config
        self.risk_scaler_train = None
        self.risk_scaler_test = None
        self._risk_quantiles_ = None
        self._risk_piecewise_levels_ = None
    
    def build(self, feature_df, train_len):
        """
        构建 Risk 缩放因子（分位数分段）
        
        Args:
            feature_df (pd.DataFrame): 特征矩阵（全样本）
            train_len (int): 训练集长度
        
        Returns:
            tuple: (risk_scaler_train, risk_scaler_test)
        """
        print("正在构建 风控 & 拥挤度 层缩放因子...")
        n = len(feature_df)
        
        def _pick_cols(substr_list):
            cols = []
            lower_cols = [c.lower() for c in feature_df.columns]
            for sub in substr_list:
                for col, lower_col in zip(feature_df.columns, lower_cols):
                    if sub in lower_col:
                        cols.append(col)
            seen = set()
            unique_cols = []
            for c in cols:
                if c not in seen:
                    seen.add(c)
                    unique_cols.append(c)
            return unique_cols
        
        # 拥挤度
        cfg_crowding = self.config.get('risk_crowding_cols')
        if cfg_crowding:
            crowding_cols = [c for c in cfg_crowding if c in feature_df.columns]
        else:
            crowding_cols = _pick_cols(['oi_zscore', 'oi_change', 'toptrader_oi_skew', 'crowd'])
        
        # 冲击
        cfg_impact = self.config.get('risk_impact_cols')
        if cfg_impact:
            impact_cols = [c for c in cfg_impact if c in feature_df.columns]
        else:
            impact_cols = _pick_cols(['amihud', 'gap_strength', 'gap_signed', 'taker_imbalance_vol'])
        
        # 资金成本
        cfg_funding = self.config.get('risk_funding_cols')
        if cfg_funding:
            funding_cols = [c for c in cfg_funding if c in feature_df.columns]
        else:
            funding_cols = _pick_cols(['funding_zscore', 'funding_rate'])
        
        risk_score = np.zeros(n, dtype=float)
        
        if crowding_cols:
            crowd_vals = feature_df[crowding_cols].values.astype(float)
            crowd_vals = np.nan_to_num(crowd_vals, nan=0.0, posinf=0.0, neginf=0.0)
            crowd_pos = np.maximum(crowd_vals, 0.0)
            risk_score += 0.5 * crowd_pos.mean(axis=1)
        
        if impact_cols:
            imp_vals = feature_df[impact_cols].values.astype(float)
            imp_vals = np.nan_to_num(imp_vals, nan=0.0, posinf=0.0, neginf=0.0)
            imp_pos = np.maximum(imp_vals, 0.0)
            risk_score += 0.3 * imp_pos.mean(axis=1)
        
        if funding_cols:
            fund_vals = feature_df[funding_cols].values.astype(float)
            fund_vals = np.nan_to_num(fund_vals, nan=0.0, posinf=0.0, neginf=0.0)
            fund_pos = np.maximum(fund_vals, 0.0)
            risk_score += 0.2 * fund_pos.mean(axis=1)
        
        risk_score = np.nan_to_num(risk_score, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 分位数分段
        risk_train = risk_score[:train_len]
        q1, q2, q3 = np.quantile(risk_train, [0.5, 0.8, 0.95])
        self._risk_quantiles_ = (q1, q2, q3)
        
        levels = (1.0, 0.7, 0.4, 0.2)
        self._risk_piecewise_levels_ = levels
        
        risk_scaler_all = self._risk_piecewise_from_quantiles(
            risk_score, q1, q2, q3, levels
        )
        
        self.risk_scaler_train = risk_scaler_all[:train_len]
        self.risk_scaler_test = risk_scaler_all[train_len:]
        
        print(f"风控 & 拥挤度 缩放构建完成（quantile-based 分段）："
              f"train_mean={self.risk_scaler_train.mean():.3f}, "
              f"test_mean={self.risk_scaler_test.mean():.3f}")
        
        return self.risk_scaler_train, self.risk_scaler_test
    
    @staticmethod
    def _risk_piecewise_from_quantiles(score_arr, q1, q2, q3, levels):
        """分位数分段映射"""
        score_arr = np.asarray(score_arr).flatten()
        f1, f2, f3, f4 = levels
        scaler = np.full_like(score_arr, f4, dtype=float)
        
        mask1 = score_arr <= q1
        mask2 = (score_arr > q1) & (score_arr <= q2)
        mask3 = (score_arr > q2) & (score_arr <= q3)
        mask4 = score_arr > q3
        
        scaler[mask1] = f1
        scaler[mask2] = f2
        scaler[mask3] = f3
        scaler[mask4] = f4
        
        return scaler


class KellyBetSizer:
    """
    Lopez 风格 Kelly bet sizing
    基于 Triple Barrier 的胜率 + 盈亏比
    """
    
    def __init__(self, config):
        self.config = config
        self.meta_model = None
        self.meta_p_train = None
        self.meta_p_test = None
        self.R = None
    
    def train_meta_model(self, X_train, X_test, meta_labels_all, train_len):
        """
        训练 meta 模型（预测胜率）
        
        Args:
            X_train (np.ndarray): 训练特征
            X_test (np.ndarray): 测试特征
            meta_labels_all (np.ndarray): Triple Barrier meta labels（1=盈利，0=亏损）
            train_len (int): 训练集长度
        
        Returns:
            self
        """
        total_len = train_len + len(X_test)
        meta_arr = np.asarray(meta_labels_all).astype(float)[:total_len]
        
        meta_train = meta_arr[:train_len]
        meta_test = meta_arr[train_len:total_len]
        
        # 仅在有 trade-level meta-label 的样本上训练（过滤 NaN）
        mask_train = np.isfinite(meta_train)
        if not np.any(mask_train):
            print("⚠️ Meta 模型训练中没有有效的 trade-level 标签，跳过 Kelly 胜率估计")
            return self
        
        X_train_sub = np.asarray(X_train)[mask_train]
        meta_train_sub = meta_train[mask_train]
        
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_sub, meta_train_sub)
        self.meta_model = clf
        
        # 在完整时间轴上预测 P(meta=1)，包括非开仓 bar（用于连续 Kelly 缩放）
        self.meta_p_train = clf.predict_proba(X_train)[:, 1]
        self.meta_p_test = clf.predict_proba(X_test)[:, 1]
        
        print("Meta 模型训练完成（用于 Kelly 胜率估计）")
        print(f"  meta=1 占比（train）：{meta_train.mean():.2%}")
        print(f"  预测 P(meta=1) 均值（train）：{self.meta_p_train.mean():.2%}")
        
        return self
    
    def compute_R_from_barrier(self, barrier_results, train_len):
        """
        计算全局盈亏比 R = avg(win) / avg(|loss|)
        
        Args:
            barrier_results (pd.DataFrame): Triple Barrier 结果
            train_len (int): 训练集长度
        
        Returns:
            float: 盈亏比 R
        """
        ret_tb = np.asarray(barrier_results['ret'].values)
        ret = ret_tb[:train_len]
        
        wins = ret[ret > 0]
        losses = ret[ret < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            print("⚠️ Kelly 计算中无 win 或 loss 样本，R=0")
            return 0.0
        
        avg_win = wins.mean()
        avg_loss = np.abs(losses.mean())
        
        if avg_loss <= 1e-8:
            print("⚠️ Kelly 计算中 avg_loss 接近 0，R=0")
            return 0.0
        
        self.R = avg_win / avg_loss
        print(f"基于 Triple Barrier 的全局盈亏比 R = {self.R:.3f}")
        return self.R
    
    def apply_kelly_sizing(self, predictions, base_model_name='Ensemble'):
        """
        应用 Kelly bet sizing 到所有模型预测
        
        Args:
            predictions (dict): 所有模型的预测结果
            base_model_name (str): 用于提取方向的基准模型
        
        Returns:
            dict: 更新后的预测结果
        """
        if self.R is None or self.R <= 0:
            print("⚠️ R <= 0，Kelly bet sizing 失效，保持原始仓位")
            return predictions
        
        c = float(self.config.get('kelly_fraction', 0.25))
        clip_num = float(self.config.get('clip_num', 5.0))
        
        p_train = np.asarray(self.meta_p_train)
        p_test = np.asarray(self.meta_p_test)
        
        # Kelly 理论比例 f* = p - (1-p)/R
        f_train = p_train - (1.0 - p_train) / self.R
        f_test = p_test - (1.0 - p_test) / self.R
        f_train = np.maximum(f_train, 0.0)
        f_test = np.maximum(f_test, 0.0)
        
        size_train = np.clip(c * f_train, 0.0, clip_num)
        size_test = np.clip(c * f_test, 0.0, clip_num)
        
        print(f"Kelly 仓位 sizing 完成：c={c}, "
              f"size_train_mean={size_train.mean():.3f}, size_test_mean={size_test.mean():.3f}")
        
        # 方向：使用基准模型
        if base_model_name not in predictions:
            base_model_name = next(iter(predictions.keys()))
            print(f"指定的基准模型不存在，改用 {base_model_name} 作为 Kelly 方向来源")
        
        base_train = np.asarray(predictions[base_model_name]['train']).flatten()
        base_test = np.asarray(predictions[base_model_name]['test']).flatten()
        
        side_train = np.sign(base_train)
        side_test = np.sign(base_test)
        side_train[np.isnan(side_train)] = 0.0
        side_test[np.isnan(side_test)] = 0.0
        
        # 应用到所有模型
        for model_name in predictions.keys():
            train_pos_new = side_train * size_train
            test_pos_new = side_test * size_test
            predictions[model_name]['train'] = train_pos_new
            predictions[model_name]['test'] = test_pos_new
        
        return predictions


class PositionScalingManager:
    """
    统一管理所有仓位缩放层（Regime + Risk + Kelly）
    """
    
    def __init__(self, config, feature_df, train_len):
        """
        Args:
            config (dict): 策略配置
            feature_df (pd.DataFrame): 特征矩阵（用于构建缩放因子）
            train_len (int): 训练集长度
        """
        self.config = config
        self.feature_df = feature_df
        self.train_len = train_len
        
        self.regime_scaler = RegimeScaler(config)
        self.risk_scaler = RiskScaler(config)
        self.kelly_sizer = KellyBetSizer(config)
        
        self.regime_scaler_train = None
        self.regime_scaler_test = None
        self.risk_scaler_train = None
        self.risk_scaler_test = None
    
    def build_regime_and_risk_scalers(self):
        """构建 Regime 和 Risk 缩放因子"""
        if self.config.get('enable_regime_layer', True):
            self.regime_scaler_train, self.regime_scaler_test = self.regime_scaler.build(
                self.feature_df, self.train_len
            )
        
        if self.config.get('enable_risk_layer', True):
            self.risk_scaler_train, self.risk_scaler_test = self.risk_scaler.build(
                self.feature_df, self.train_len
            )
        
        return self
    
    def apply_to_predictions(self, predictions):
        """
        将 Regime & Risk 缩放应用到预测仓位
        
        Args:
            predictions (dict): 模型预测结果
        
        Returns:
            dict: 缩放后的预测结果
        """
        use_regime = self.config.get('enable_regime_layer', True)
        use_risk = self.config.get('enable_risk_layer', True)
        
        if not use_regime and not use_risk:
            print("配置中关闭了 Regime 层与风控层缩放，保持原始模型仓位")
            return predictions
        
        print("开始应用 Regime 层与 风控 & 拥挤度 层缩放...")
        
        any_model = next(iter(predictions.values()))
        base_train_len = len(any_model['train'])
        base_test_len = len(any_model['test'])
        
        train_scaler = np.ones(base_train_len, dtype=float)
        test_scaler = np.ones(base_test_len, dtype=float)
        
        if use_regime and self.regime_scaler_train is not None:
            train_scaler *= self._align_and_expand(self.regime_scaler_train, base_train_len)
            test_scaler *= self._align_and_expand(self.regime_scaler_test, base_test_len)
        
        if use_risk and self.risk_scaler_train is not None:
            train_scaler *= self._align_and_expand(self.risk_scaler_train, base_train_len)
            test_scaler *= self._align_and_expand(self.risk_scaler_test, base_test_len)
        
        for model_name, pred in predictions.items():
            train_pos = np.asarray(pred['train']).flatten()
            test_pos = np.asarray(pred['test']).flatten()
            
            predictions[model_name]['train'] = train_pos * train_scaler
            predictions[model_name]['test'] = test_pos * test_scaler
        
        print("三层结构缩放应用完成（Alpha → Regime → 风控 & 拥挤度）")
        return predictions
    
    @staticmethod
    def _align_and_expand(scaler, target_len):
        """对齐缩放数组到目标长度"""
        scaler = np.asarray(scaler).flatten()
        if len(scaler) >= target_len:
            return scaler[:target_len]
        if len(scaler) == 0:
            return np.ones(target_len, dtype=float)
        pad_len = target_len - len(scaler)
        return np.concatenate([scaler, np.full(pad_len, scaler[-1], dtype=float)])
