"""
诊断工具模块
负责 Label 健康度、因子 IC、单因子回测等诊断功能
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class DiagnosticTools:
    """
    诊断工具集：Label 健康度、因子 IC、单因子回测
    """
    
    def __init__(self, factor_data, selected_factors, ret_train, ret_test, 
                 open_train, close_train, open_test, close_test, fees_rate, annual_bars):
        """
        Args:
            factor_data (pd.DataFrame): 因子数据
            selected_factors (list): 选中的因子列表
            ret_train (np.ndarray): 训练标签
            ret_test (np.ndarray): 测试标签
            open_train/close_train: 训练集价格
            open_test/close_test: 测试集价格
            fees_rate (float): 手续费率
            annual_bars (int): 年化 bar 数
        """
        self.factor_data = factor_data
        self.selected_factors = selected_factors
        self.ret_train = ret_train
        self.ret_test = ret_test
        self.open_train = open_train
        self.close_train = close_train
        self.open_test = open_test
        self.close_test = close_test
        self.fees_rate = fees_rate
        self.annual_bars = annual_bars
    
    def diagnose_label_health(self, barrier_results=None, meta_labels=None):
        """
        检查 label 的健康度（分布、正负样本占比）
        
        Args:
            barrier_results (pd.DataFrame, optional): Triple Barrier 结果
            meta_labels (np.ndarray, optional): Meta labels
        """
        print("\n===== Label 健康度诊断 =====")
        
        ret_train = np.asarray(self.ret_train).flatten()
        ret_test = np.asarray(self.ret_test).flatten()
        
        def _summary(arr, name):
            arr = np.asarray(arr)
            print(f"\n[{name}]")
            print(f"  样本数: {len(arr)}")
            if len(arr) == 0:
                return
            print(f"  均值: {np.nanmean(arr):.6f}, 标准差: {np.nanstd(arr):.6f}")
            print(f"  min: {np.nanmin(arr):.6f}, max: {np.nanmax(arr):.6f}")
            pos_ratio = np.sum(arr > 0) / len(arr)
            zero_ratio = np.sum(arr == 0) / len(arr)
            neg_ratio = np.sum(arr < 0) / len(arr)
            print(f"  >0 占比: {pos_ratio:.2%}, =0 占比: {zero_ratio:.2%}, <0 占比: {neg_ratio:.2%}")
        
        _summary(ret_train, "Train Label (ret_train)")
        _summary(ret_test, "Test Label (ret_test)")
        
        # Triple Barrier 统计
        if barrier_results is not None:
            tb_ret = np.asarray(barrier_results['ret'].values)
            print("\n[Triple Barrier 收益（全样本）]")
            _summary(tb_ret, "TB ret (all)")
        
        if meta_labels is not None:
            meta = np.asarray(meta_labels).astype(float)
            print("\n[Triple Barrier Meta Label（1=盈利,0=亏损）]")
            ones_ratio = np.mean(meta == 1)
            zeros_ratio = np.mean(meta == 0)
            print(f"  样本数: {len(meta)}")
            print(f"  meta=1 占比: {ones_ratio:.2%}, meta=0 占比: {zeros_ratio:.2%}")
        
        print("===== Label 健康度诊断结束 =====\n")
    
    def diagnose_factor_ic(self, data_range='train', top_n=20):
        """
        计算因子 IC 与 RankIC，按 |IC| 排序
        
        Args:
            data_range (str): 'train' 或 'test'
            top_n (int): 输出前 N 个因子
        
        Returns:
            pd.DataFrame: IC 诊断结果
        """
        print("\n===== 因子 IC / RankIC 诊断 =====")
        
        train_len = len(self.ret_train)
        test_len = len(self.ret_test)
        
        if data_range == 'train':
            # 训练段：取前 train_len 行，与 ret_train 对齐
            fac_df = self.factor_data[self.selected_factors].iloc[:train_len]
            y = np.asarray(self.ret_train).flatten()[:len(fac_df)]
            print("使用训练集 ret_train 作为 IC 计算的目标。")
        elif data_range == 'test':
            # 测试段：从尾部取 test_len 行，与 ret_test 对齐
            fac_df = self.factor_data[self.selected_factors].iloc[-test_len:]
            y = np.asarray(self.ret_test).flatten()[:len(fac_df)]
            print("使用测试集 ret_test 作为 IC 计算的目标。")
        else:
            raise ValueError("data_range 必须是 'train' 或 'test'")
        
        records = []
        y = np.asarray(y)
        for col in fac_df.columns:
            x = fac_df[col].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 50:
                continue
            x_valid = x[mask]
            y_valid = y[mask]
            
            # 皮尔逊 IC
            try:
                ic = np.corrcoef(x_valid, y_valid)[0, 1]
            except Exception:
                ic = np.nan
            
            # Spearman RankIC
            try:
                rk = spearmanr(x_valid, y_valid).correlation
            except Exception:
                rk = np.nan
            
            records.append((col, ic, rk))
        
        if not records:
            print("有效因子样本不足，无法计算 IC。")
            return None
        
        df_ic = pd.DataFrame(records, columns=['factor', 'IC', 'RankIC'])
        df_ic['|IC|'] = df_ic['IC'].abs()
        df_ic = df_ic.sort_values('|IC|', ascending=False)
        
        print(f"\n按 |IC| 排序的前 {min(top_n, len(df_ic))} 个因子：")
        print(df_ic.head(top_n).to_string(index=False, float_format=lambda x: f"{x: .4f}"))
        
        print("===== 因子 IC / RankIC 诊断结束 =====\n")
        return df_ic
    
    def backtest_single_factor(self, factor_name, data_range='test', n_quantiles=5):
        """
        单因子多空分层回测
        
        Args:
            factor_name (str): 因子名称
            data_range (str): 'train' 或 'test'
            n_quantiles (int): 分位数数量
        
        Returns:
            tuple: (pnl, metrics)
        """
        if factor_name not in self.factor_data.columns:
            print(f"因子 {factor_name} 不在 factor_data 中。")
            return None
        
        train_len = len(self.ret_train)
        if data_range == 'train':
            fac_vals = self.factor_data[factor_name].iloc[:train_len].values
            pos = self._build_long_short_position(fac_vals, n_quantiles)
            pnl, metrics = self._simulate_trading(pos, 'train')
        elif data_range == 'test':
            fac_vals = self.factor_data[factor_name].iloc[train_len:].values
            pos = self._build_long_short_position(fac_vals, n_quantiles)
            pnl, metrics = self._simulate_trading(pos, 'test')
        else:
            raise ValueError("data_range 必须是 'train' 或 'test'")
        
        print(f"\n===== 单因子多空回测：{factor_name} | {data_range} 段 =====")
        for k, v in metrics.items():
            if "Rate" in k or "Ratio" in k or "Return" in k or "Drawdown" in k:
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print("===== 单因子多空回测结束 =====\n")
        
        return pnl, metrics
    
    def diagnose_top_factors_backtest(self, data_range='test', top_n=5, n_quantiles=5):
        """
        对 |IC| Top N 因子做单因子回测
        
        Args:
            data_range (str): 'train' 或 'test'
            top_n (int): Top N 个因子
            n_quantiles (int): 分位数数量
        
        Returns:
            dict: 回测结果
        """
        df_ic = self.diagnose_factor_ic(data_range=data_range, top_n=top_n)
        if df_ic is None or df_ic.empty:
            return None
        
        top_factors = df_ic['factor'].head(top_n).tolist()
        results = {}
        print(f"\n>>> 对 |IC| Top {len(top_factors)} 因子做单因子多空回测（{data_range} 段）")
        for fct in top_factors:
            _, metrics = self.backtest_single_factor(
                factor_name=fct,
                data_range=data_range,
                n_quantiles=n_quantiles
            )
            results[fct] = metrics
        
        return results
    
    def diagnose_factor_rolling_ic(self, factor_name, data_range='train', window=500, method='spearman'):
        """
        计算单个因子的滚动 IC 序列，用于观察时间稳定性。
        
        Args:
            factor_name (str): 因子名称
            data_range (str): 'train' 或 'test'
            window (int): 滚动窗口长度（bar 数）
            method (str): 'pearson' 或 'spearman'
        Returns:
            pd.Series: index 与对应区间的时间索引对齐的滚动 IC 序列
        """
        print(f"\n===== 因子滚动 IC 诊断：{factor_name} | {data_range} 段 | window={window} =====")
        if factor_name not in self.factor_data.columns:
            print(f"因子 {factor_name} 不在 factor_data 中。")
            return None
        
        train_len = len(self.ret_train)
        if data_range == 'train':
            fac_series = self.factor_data[factor_name].iloc[:train_len]
            y = np.asarray(self.ret_train).flatten()[:len(fac_series)]
        elif data_range == 'test':
            fac_series = self.factor_data[factor_name].iloc[train_len:]
            y = np.asarray(self.ret_test).flatten()[:len(fac_series)]
        else:
            raise ValueError("data_range 必须是 'train' 或 'test'")
        
        y_series = pd.Series(y, index=fac_series.index)
        # 统一清理
        df = pd.DataFrame({"x": fac_series, "y": y_series}).replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < window:
            print("有效样本不足以计算滚动 IC。")
            return None
        
        if method == 'pearson':
            roll_ic = df["x"].rolling(window=window).corr(df["y"])
        else:
            # 近似 Spearman：对全样本做 rank，再做 Pearson rolling corr
            x_rank = df["x"].rank(method="average")
            y_rank = df["y"].rank(method="average")
            roll_ic = x_rank.rolling(window=window).corr(y_rank)
        
        print(f"滚动 IC 描述统计（去除 NaN 后）：")
        desc = roll_ic.dropna().describe()
        print(desc.to_string(float_format=lambda v: f"{v: .4f}"))
        print("===== 因子滚动 IC 诊断结束 =====\n")
        return roll_ic
    
    def diagnose_factor_ic_decay(self, horizons=(1, 3, 5), data_range='train'):
        """
        基于收盘价构造未来多周期收益，计算因子 IC Decay（时序单因子常用）。
        
        市场假设：
            当前因子值对未来 h 根 K 线的累计 log 收益有线性 / 单调关系。
        
        Args:
            horizons (tuple or list): 持有期列表，例如 (1, 3, 5)
            data_range (str): 'train' 或 'test'
        Returns:
            dict: {factor_name: {h: {'IC': x, 'RankIC': y}}}
        """
        print("\n===== 因子 IC Decay 诊断 =====")
        train_len = len(self.ret_train)
        
        # 选择对应区间的因子和收盘价
        if data_range == 'train':
            fac_df = self.factor_data[self.selected_factors].iloc[:train_len].copy()
            close_data = self.close_train
        elif data_range == 'test':
            fac_df = self.factor_data[self.selected_factors].iloc[train_len:].copy()
            close_data = self.close_test
        else:
            raise ValueError("data_range 必须是 'train' 或 'test'")
        
        if isinstance(close_data, pd.Series):
            close_arr = close_data.values
        else:
            close_arr = np.asarray(close_data)
        
        # 与因子数据长度对齐
        close_arr = close_arr[:len(fac_df)]
        if len(close_arr) != len(fac_df):
            print("警告：收盘价长度与因子长度不一致，已截断对齐。")
        
        if len(close_arr) < 3:
            print("样本太短，无法进行 IC Decay 诊断。")
            return {}
        
        # 构造单步 log 收益
        log_price = np.log(close_arr.astype(float))
        ret1 = np.concatenate([[0.0], np.diff(log_price)])  # 与价格长度对齐
        ret_series = pd.Series(ret1)
        
        # 预先为每个 horizon 构造未来累计收益（对齐到当前时刻）
        fwd_ret_dict = {}
        for h in horizons:
            if h <= 0:
                continue
            # r_{t:t+h-1} 的累计和，对齐到 t
            fwd = ret_series.rolling(window=h, min_periods=h).sum().shift(-(h-1))
            fwd_ret_dict[h] = fwd.values
        
        results = {}
        for fct in self.selected_factors:
            if fct not in fac_df.columns:
                continue
            x_all = fac_df[fct].values.astype(float)
            res_h = {}
            for h, y_all in fwd_ret_dict.items():
                if len(y_all) != len(x_all):
                    m = min(len(y_all), len(x_all))
                    x = x_all[:m]
                    y = y_all[:m]
                else:
                    x, y = x_all, y_all
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 50:
                    continue
                x_valid = x[mask]
                y_valid = y[mask]
                
                # 皮尔逊 IC
                try:
                    ic = np.corrcoef(x_valid, y_valid)[0, 1]
                except Exception:
                    ic = np.nan
                
                # Spearman RankIC
                try:
                    rk = spearmanr(x_valid, y_valid).correlation
                except Exception:
                    rk = np.nan
                
                res_h[h] = {"IC": float(ic), "RankIC": float(rk)}
            
            if res_h:
                results[fct] = res_h
        
        print("===== 因子 IC Decay 诊断结束（返回 dict，可用于画图或进一步分析） =====\n")
        return results
    
    def diagnose_factor_quantile_returns(self, factor_name, data_range='test', n_quantiles=5):
        """
        按因子分位数统计未来收益，用于检查分层单调性和极值组表现。
        
        Args:
            factor_name (str): 因子名称
            data_range (str): 'train' 或 'test'
            n_quantiles (int): 分位数数量
        Returns:
            pd.DataFrame: index=分位数, columns=[mean, std, count]
        """
        print(f"\n===== 因子分位数收益诊断：{factor_name} | {data_range} 段 =====")
        if factor_name not in self.factor_data.columns:
            print(f"因子 {factor_name} 不在 factor_data 中。")
            return None
        
        train_len = len(self.ret_train)
        if data_range == 'train':
            fac_vals = self.factor_data[factor_name].iloc[:train_len]
            y = np.asarray(self.ret_train).flatten()[:len(fac_vals)]
        elif data_range == 'test':
            fac_vals = self.factor_data[factor_name].iloc[train_len:]
            y = np.asarray(self.ret_test).flatten()[:len(fac_vals)]
        else:
            raise ValueError("data_range 必须是 'train' 或 'test'")
        
        df = pd.DataFrame({"factor": fac_vals, "ret": y})
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if df.empty:
            print("有效样本为空，无法做分位数诊断。")
            return None
        
        try:
            q = pd.qcut(df["factor"].rank(method="first"), q=n_quantiles,
                        labels=False, duplicates="drop")
        except ValueError:
            print("分位数划分失败，可能因子取值过少或重复严重。")
            return None
        
        df["q"] = q
        stats = df.groupby("q")["ret"].agg(["mean", "std", "count"])
        print(stats.to_string(float_format=lambda x: f"{x: .6f}"))
        print("===== 因子分位数收益诊断结束 =====\n")
        return stats
    
    def diagnose_factor_correlation(self, corr_threshold=0.9):
        """
        对 selected_factors 做相关矩阵，并打印高相关因子对，用于冗余因子筛选。
        
        Args:
            corr_threshold (float): |corr| 大于该阈值视为高度相关
        Returns:
            pd.DataFrame: 因子相关矩阵（Spearman）
        """
        print("\n===== 因子相关性诊断 =====")
        fac_df = self.factor_data[self.selected_factors].copy()
        fac_df = fac_df.replace([np.inf, -np.inf], np.nan).dropna(how="all")
        if fac_df.empty:
            print("有效因子数据为空，无法计算相关性。")
            return None
        
        corr = fac_df.corr(method="spearman")
        high_pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c = corr.iloc[i, j]
                if abs(c) >= corr_threshold:
                    high_pairs.append((cols[i], cols[j], c))
        
        if high_pairs:
            print(f"高相关因子对（|corr| >= {corr_threshold:.2f}）：")
            for a, b, c in high_pairs:
                print(f"  {a} - {b}: {c:.4f}")
        else:
            print(f"无 |corr| >= {corr_threshold:.2f} 的因子对。")
        
        print("===== 因子相关性诊断结束 =====\n")
        return corr
    
    def run_full_diagnostics(
        self,
        data_range_main='test',
        top_n_ic=20,
        top_n_backtest=5,
        n_quantiles=5,
        horizons_ic_decay=(1, 3, 5),
        corr_threshold=0.9,
    ):
        """
        一键运行常用因子诊断流程，便于快速评估一批因子。
        
        流程：
            1) Label 健康度（全样本）
            2) 训练 / 测试 IC 排序
            3) IC Decay（使用 data_range_main）
            4) |IC| TopN 因子单因子多空回测（data_range_main）
            5) 因子间相关性诊断
        
        Args:
            data_range_main (str): 'train' 或 'test'，决定 IC Decay 与单因子回测所在区间
            top_n_ic (int): IC 报表中展示的 Top N
            top_n_backtest (int): 做单因子回测的因子数量（基于 |IC| Top N）
            n_quantiles (int): 多空分层的分位数数量
            horizons_ic_decay (tuple): IC Decay 的持有期列表
            corr_threshold (float): 高相关判定阈值
        Returns:
            dict: 各模块结果字典
        """
        print("\n===== 开始一键因子诊断（run_full_diagnostics） =====")
        results = {}
        
        # 1) Label 健康度（全样本）
        self.diagnose_label_health()
        
        # 2) 训练 / 测试 IC 排序
        df_ic_train = self.diagnose_factor_ic(data_range='train', top_n=top_n_ic)
        df_ic_test = self.diagnose_factor_ic(data_range='test', top_n=top_n_ic)
        results["ic_train"] = df_ic_train
        results["ic_test"] = df_ic_test
        
        # 3) IC Decay（主要看 data_range_main）
        ic_decay = self.diagnose_factor_ic_decay(horizons=horizons_ic_decay, data_range=data_range_main)
        results["ic_decay"] = ic_decay
        
        # 4) |IC| TopN 单因子回测（data_range_main）
        backtest_top = self.diagnose_top_factors_backtest(
            data_range=data_range_main,
            top_n=top_n_backtest,
            n_quantiles=n_quantiles,
        )
        results["backtest_top"] = backtest_top
        
        # 5) 因子间相关性诊断
        corr_mat = self.diagnose_factor_correlation(corr_threshold=corr_threshold)
        results["corr"] = corr_mat
        
        print("===== 一键因子诊断结束（run_full_diagnostics） =====\n")
        return results
    
    def _build_long_short_position(self, factor_values, n_quantiles=5):
        """构建多空分层仓位（顶层+1，底层-1）"""
        factor_values = np.asarray(factor_values).astype(float)
        if len(factor_values) == 0:
            return np.array([], dtype=float)
        
        s = pd.Series(factor_values)
        if s.nunique() <= 1:
            return np.zeros(len(s), dtype=float)
        
        try:
            q = pd.qcut(s.rank(method='first'), q=n_quantiles, labels=False, duplicates='drop')
        except ValueError:
            return np.zeros(len(s), dtype=float)
        
        pos = np.zeros(len(s), dtype=float)
        if q.max() == q.min():
            return pos
        pos[q == q.max()] = 1.0
        pos[q == q.min()] = -1.0
        return pos
    
    def _simulate_trading(self, pos, data_range):
        """简化的交易模拟（复用 BacktestEngine 逻辑）"""
        if data_range == 'train':
            open_data = self.open_train
            close_data = self.close_train
        else:
            open_data = self.open_test
            close_data = self.close_test
        
        if isinstance(open_data, pd.Series):
            open_data = open_data.values
        if isinstance(close_data, pd.Series):
            close_data = close_data.values
        
        pos = np.asarray(pos).flatten()
        min_len = min(len(pos), len(open_data), len(close_data))
        pos = pos[:min_len]
        open_data = open_data[:min_len]
        close_data = close_data[:min_len]
        
        next_open = np.concatenate((open_data[1:], np.array([close_data[-1]])))
        pos_change = np.concatenate((np.array([0]), np.diff(pos)))
        
        which_price = np.where(
            pos_change > 0,
            np.maximum(close_data, next_open),
            np.where(pos_change < 0, np.minimum(close_data, next_open), close_data)
        )
        
        next_trade_close = np.concatenate((which_price[1:], np.array([which_price[-1]])))
        rets = np.log(next_trade_close) - np.log(which_price)
        
        gain_loss = pos * rets - abs(pos_change) * self.fees_rate
        pnl = gain_loss.cumsum()
        
        # 计算指标
        win_rate = np.sum(gain_loss > 0) / len(gain_loss) if len(gain_loss) > 0 else 0
        avg_gain = np.mean(gain_loss[gain_loss > 0]) if np.any(gain_loss > 0) else 0
        avg_loss = np.abs(np.mean(gain_loss[gain_loss < 0])) if np.any(gain_loss < 0) else 0
        pl_ratio = avg_gain / avg_loss if avg_loss != 0 else np.inf
        
        annual_return = np.mean(gain_loss) * self.annual_bars
        sharpe = annual_return / (np.std(gain_loss) * np.sqrt(self.annual_bars)) if np.std(gain_loss) > 0 else 0
        
        peak = np.maximum.accumulate(pnl)
        peak = np.where(peak == 0, 1.0, peak)
        dd = (pnl - peak) / np.abs(peak)
        max_dd = np.min(dd) if len(dd) > 0 else 0
        
        calmar = annual_return / abs(max_dd) if max_dd < -0.0001 else (np.inf if annual_return > 0 else 0)
        
        turnover = np.mean(np.abs(pos_change)) if len(pos_change) > 0 else 0.0
        avg_pos = np.mean(np.abs(pos)) if len(pos) > 0 else 0.0
        
        metrics = {
            "Win Rate": win_rate,
            "Profit/Loss Ratio": pl_ratio,
            "Annual Return": annual_return,
            "MAX_Drawdown": max_dd,
            "Sharpe Ratio": sharpe,
            "Calmar Ratio": calmar,
            "Turnover_per_bar": turnover,
            "Avg_abs_position": avg_pos
        }
        
        return pnl, metrics

