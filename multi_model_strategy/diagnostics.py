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
        if data_range == 'train':
            fac_df = self.factor_data[self.selected_factors].iloc[:train_len]
            y = np.asarray(self.ret_train).flatten()[:len(fac_df)]
            print("使用训练集 ret_train 作为 IC 计算的目标。")
        elif data_range == 'test':
            fac_df = self.factor_data[self.selected_factors].iloc[train_len:]
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
        
        metrics = {
            "Win Rate": win_rate,
            "Profit/Loss Ratio": pl_ratio,
            "Annual Return": annual_return,
            "MAX_Drawdown": max_dd,
            "Sharpe Ratio": sharpe,
            "Calmar Ratio": calmar
        }
        
        return pnl, metrics

