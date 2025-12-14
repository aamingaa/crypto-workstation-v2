"""
诊断工具模块
负责 Label 健康度、因子 IC、单因子回测等诊断功能
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


class DiagnosticTools:
    """
    诊断工具集：Label 健康度、因子 IC、单因子回测
    """
    
    def __init__(self, factor_data, selected_factors, ret_train, ret_test, 
                 y_train, y_test, open_train, close_train, open_test, close_test, fees_rate, annual_bars):
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
        self.y_train = y_train
        self.y_test = y_test
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
        y_train = np.asarray(self.y_train).flatten()
        y_test = np.asarray(self.y_test).flatten()
        
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
        _summary(y_train, "Train Label (y_train)")
        _summary(y_test, "Test Label (y_test)")
        
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
        test_len = len(self.ret_test)
        if data_range == 'train':
            fac_vals = self.factor_data[factor_name].iloc[:train_len].values
            pos = self._build_long_short_position(fac_vals, n_quantiles)
            pnl, metrics = self._simulate_trading(pos, 'train')
        elif data_range == 'test':
            fac_vals = self.factor_data[factor_name].iloc[-test_len:].values
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
    
    def backtest_single_factor_by_quantile_weights(self, factor_name, weights, data_range='test', n_quantiles=5):
        """
        按自定义分箱权重做单因子回测（例如：{0:-1, 3:0.5, 4:1}）。
        
        市场假设：
            因子在不同分箱上的 alpha 能力不同，只在部分分箱交易或赋予不同仓位权重。
        
        Args:
            factor_name (str): 因子名称
            weights (dict): {quantile_index: weight}，quantile_index 从 0 到 n_quantiles-1
            data_range (str): 'train' 或 'test'
            n_quantiles (int): 分位数数量
        
        Returns:
            tuple: (pnl, metrics)
        """
        if factor_name not in self.factor_data.columns:
            print(f"因子 {factor_name} 不在 factor_data 中。")
            return None
        
        train_len = len(self.ret_train)
        test_len = len(self.ret_test)
        if data_range == 'train':
            fac_vals = self.factor_data[factor_name].iloc[:train_len].values
            pos = self._build_position_by_quantile_weights(fac_vals, weights, n_quantiles)
            pnl, metrics = self._simulate_trading(pos, 'train')
        elif data_range == 'test':
            fac_vals = self.factor_data[factor_name].iloc[-test_len:].values
            pos = self._build_position_by_quantile_weights(fac_vals, weights, n_quantiles)
            pnl, metrics = self._simulate_trading(pos, 'test')
        else:
            raise ValueError("data_range 必须是 'train' 或 'test'")
        
        print(f"\n===== 单因子分箱权重回测：{factor_name} | {data_range} 段 =====")
        print(f"  使用分箱权重: {weights}")
        for k, v in metrics.items():
            if "Rate" in k or "Ratio" in k or "Return" in k or "Drawdown" in k:
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print("===== 单因子分箱权重回测结束 =====\n")
        
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
    
    def diagnose_factor_quantile_returns(self, factor_name, data_range='test', n_quantiles=5, save_dir=None):
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
        
        # 绘图：每个分位数的平均收益柱状图
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            stats_plot = stats.copy()
            stats_plot["mean"].plot(kind="bar", ax=ax)
            ax.set_title(f"Quantile Returns - {factor_name} | {data_range}")
            ax.set_xlabel("Quantile (0 = lowest)")
            ax.set_ylabel("Mean future return")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
            plt.tight_layout()
            
            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                fname = f"quantile_returns_{factor_name}_{data_range}.png"
                # 简单清理文件名中的斜杠等
                fname = fname.replace("/", "_").replace("\\", "_").replace(" ", "_")
                (save_dir / fname).unlink(missing_ok=True)
                plt.savefig(save_dir / fname, dpi=150)
                plt.close(fig)
            else:
                plt.show()
        except Exception as e:
            print(f"绘制分位数收益图失败: {e}")
        
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
        top_n_ic=10,
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
    
    def save_full_diagnostics_to_dir(
        self,
        results: dict,
        save_dir,
        title: str = None,
        max_ic_bars: int = 10,
        max_decay_factors: int = 5,
        max_sharpe_bars: int = 10,
        max_corr_factors: int = 20,
    ):
        """
        将 run_full_diagnostics 的结果落盘（csv + 综览大图）。
        
        Args:
            results (dict): run_full_diagnostics 的返回结果
            save_dir (str or Path): 输出目录
            title (str): 综览图标题
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        ic_train = results.get("ic_train")
        ic_test = results.get("ic_test")
        ic_decay = results.get("ic_decay", {})
        backtest_top = results.get("backtest_top", {})
        corr_mat = results.get("corr")

        # 1) IC 报表
        if isinstance(ic_train, pd.DataFrame):
            ic_train.to_csv(save_dir / "ic_train.csv", index=False)
        if isinstance(ic_test, pd.DataFrame):
            ic_test.to_csv(save_dir / "ic_test.csv", index=False)

        # 2) IC Decay 长表
        decay_records = []
        for fct, h_dict in ic_decay.items():
            for h, vals in h_dict.items():
                decay_records.append({
                    "factor": fct,
                    "horizon": h,
                    "IC": vals.get("IC", np.nan),
                    "RankIC": vals.get("RankIC", np.nan),
                })
        if decay_records:
            df_decay = pd.DataFrame(decay_records)
            df_decay.sort_values(["factor", "horizon"], inplace=True)
            df_decay.to_csv(save_dir / "ic_decay_long.csv", index=False)

        # 3) 单因子回测指标
        if backtest_top:
            df_bt = pd.DataFrame(backtest_top).T
            df_bt.index.name = "factor"
            df_bt.reset_index(inplace=True)
            df_bt.to_csv(save_dir / "backtest_top_factors.csv", index=False)
        else:
            df_bt = None

        # 4) 因子相关矩阵
        if isinstance(corr_mat, pd.DataFrame):
            corr_mat.to_csv(save_dir / "factor_corr.csv", index=True)

        # 5) 单独绘制若干诊断图 + 可选：按分箱权重的策略回测
        try:
            # (1) 测试集 |IC| TopK 柱状图
            if isinstance(ic_test, pd.DataFrame) and not ic_test.empty:
                fig_ic, ax_ic = plt.subplots(figsize=(10, 4))
                df_plot_ic = ic_test.copy()
                sort_col = "|IC|" if "|IC|" in df_plot_ic.columns else "IC"
                df_plot_ic = df_plot_ic.sort_values(sort_col, ascending=False)
                top_k = min(max_ic_bars, len(df_plot_ic))
                df_plot_ic = df_plot_ic.head(top_k)
                ax_ic.bar(df_plot_ic["factor"], df_plot_ic["IC"])
                ax_ic.set_title("Test: 单因子 IC（Top）")
                ax_ic.set_ylabel("IC")
                ax_ic.set_xticklabels(df_plot_ic["factor"], rotation=45, ha="right", fontsize=8)
                ax_ic.axhline(0, color="black", linewidth=0.8)
                ax_ic.grid(True, axis="y", linestyle="--", alpha=0.4)
                plt.tight_layout()
                (save_dir / "ic_test_top.png").unlink(missing_ok=True)
                plt.savefig(save_dir / "ic_test_top.png", dpi=150)
                plt.close(fig_ic)

            # (2) IC Decay 曲线（选部分因子）
            if ic_decay:
                fig_decay, ax_decay = plt.subplots(figsize=(10, 4))
                factors_all = list(ic_decay.keys())
                if isinstance(ic_test, pd.DataFrame) and not ic_test.empty:
                    ordered = ic_test.sort_values(
                        "|IC|" if "|IC|" in ic_test.columns else "IC",
                        ascending=False,
                    )["factor"].tolist()
                    chosen = [f for f in ordered if f in ic_decay][:max_decay_factors]
                    if not chosen:
                        chosen = factors_all[:max_decay_factors]
                else:
                    chosen = factors_all[:max_decay_factors]

                for fct in chosen:
                    h_dict = ic_decay.get(fct, {})
                    if not h_dict:
                        continue
                    hs = sorted(h_dict.keys())
                    ic_vals = [h_dict[h]["IC"] for h in hs]
                    ax_decay.plot(hs, ic_vals, marker="o", label=fct)
                ax_decay.set_title("IC Decay（部分因子）")
                ax_decay.set_xlabel("horizon")
                ax_decay.set_ylabel("IC")
                ax_decay.axhline(0, color="black", linewidth=0.8)
                ax_decay.grid(True, linestyle="--", alpha=0.4)
                if chosen:
                    ax_decay.legend(fontsize=8)
                plt.tight_layout()
                (save_dir / "ic_decay_selected.png").unlink(missing_ok=True)
                plt.savefig(save_dir / "ic_decay_selected.png", dpi=150)
                plt.close(fig_decay)

            # (3) 单因子回测 Sharpe 柱状图
            if isinstance(df_bt, pd.DataFrame) and not df_bt.empty and "Sharpe Ratio" in df_bt.columns:
                fig_bt, ax_bt = plt.subplots(figsize=(10, 4))
                df_plot_bt = df_bt.sort_values("Sharpe Ratio", ascending=False)
                top_k_bt = min(max_sharpe_bars, len(df_plot_bt))
                df_plot_bt = df_plot_bt.head(top_k_bt)
                ax_bt.bar(df_plot_bt["factor"], df_plot_bt["Sharpe Ratio"])
                ax_bt.set_title("Test: 单因子多空回测 Sharpe（Top）")
                ax_bt.set_ylabel("Sharpe")
                ax_bt.set_xticklabels(df_plot_bt["factor"], rotation=45, ha="right", fontsize=8)
                ax_bt.axhline(0, color="black", linewidth=0.8)
                ax_bt.grid(True, axis="y", linestyle="--", alpha=0.4)
                plt.tight_layout()
                (save_dir / "backtest_sharpe_top.png").unlink(missing_ok=True)
                plt.savefig(save_dir / "backtest_sharpe_top.png", dpi=150)
                plt.close(fig_bt)

            # (4) 因子相关性热力图
            if isinstance(corr_mat, pd.DataFrame) and not corr_mat.empty:
                fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                cols = corr_mat.columns.tolist()[:max_corr_factors]
                sub_corr = corr_mat.loc[cols, cols]
                im = ax_corr.imshow(sub_corr.values, cmap="coolwarm", vmin=-1, vmax=1)
                ax_corr.set_title("因子相关性热力图（Spearman）")
                ax_corr.set_xticks(range(len(cols)))
                ax_corr.set_yticks(range(len(cols)))
                ax_corr.set_xticklabels(cols, rotation=90, fontsize=7)
                ax_corr.set_yticklabels(cols, fontsize=7)
                fig_corr.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
                plt.tight_layout()
                (save_dir / "factor_corr_heatmap.png").unlink(missing_ok=True)
                plt.savefig(save_dir / "factor_corr_heatmap.png", dpi=150)
                plt.close(fig_corr)
        except Exception as e:
            print(f"绘制因子诊断综览图失败: {e}")
    
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
    
    def _build_position_by_quantile_weights(self, factor_values, weights, n_quantiles=5):
        """
        按自定义分箱权重构建仓位。
        
        Args:
            factor_values (array-like): 因子取值
            weights (dict): {quantile_index: weight}，quantile_index 从 0 到 n_quantiles-1
        Returns:
            np.ndarray: 仓位序列
        """
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
        # quantile_index 范围为 [0, n_quantiles-1]，0 为最低分箱
        for q_idx, w in weights.items():
            try:
                q_idx_int = int(q_idx)
            except Exception:
                continue
            if q_idx_int < 0 or q_idx_int > q.max():
                continue
            pos[q == q_idx_int] = float(w)
        return pos

    def build_quantile_weighted_signal(self, factor_name, weights, n_quantiles=5):
        """
        基于分箱权重，为指定因子生成一条完整的 signal 序列（train+test 拼在一起）。
        
        Args:
            factor_name (str): 因子名称
            weights (dict): {quantile_index: weight}，quantile_index 从 0 到 n_quantiles-1
            n_quantiles (int): 分箱数量
        
        Returns:
            pd.Series: index 与 factor_data 对齐的 signal（长度与 factor_data 相同）
        """
        if factor_name not in self.factor_data.columns:
            raise ValueError(f"因子 {factor_name} 不在 factor_data 中。")
        
        fac_vals_all = self.factor_data[factor_name].values
        pos_all = self._build_position_by_quantile_weights(
            factor_values=fac_vals_all,
            weights=weights,
            n_quantiles=n_quantiles,
        )
        return pd.Series(pos_all, index=self.factor_data.index, name=f"{factor_name}_qw")
    
    def visualize_factor_vs_price(
        self, 
        factor_name, 
        data_range='test', 
        price_type='close', 
        save_dir=None,
        figsize=(14, 8)
    ):
        """
        可视化单因子与价格的时序对比，用于观察因子行为与价格变动的关系。
        
        市场假设：
            某些因子在价格拐点前后会出现特征性信号（如尖峰、突变等）。
            例如：价格见顶下跌前，清算压力因子（Liq_Zscore）可能出现尖峰。
        
        Args:
            factor_name (str): 因子名称
            data_range (str): 'train' 或 'test'
            price_type (str): 'close' 或 'open'，选择要展示的价格类型
            save_dir (str or Path): 保存目录，None 则显示图形
            figsize (tuple): 图形尺寸
        
        Returns:
            fig, (ax1, ax2): matplotlib 图形对象
        """
        print(f"\n===== 单因子与价格可视化对比：{factor_name} | {data_range} 段 =====")
        
        if factor_name not in self.factor_data.columns:
            print(f"因子 {factor_name} 不在 factor_data 中。")
            return None
        
        train_len = len(self.ret_train)
        
        # 获取因子数据和价格数据
        if data_range == 'train':
            fac_series = self.factor_data[factor_name].iloc[:train_len]
            if price_type == 'close':
                price_data = self.close_train
            else:
                price_data = self.open_train
        elif data_range == 'test':
            fac_series = self.factor_data[factor_name].iloc[train_len:]
            if price_type == 'close':
                price_data = self.close_test
            else:
                price_data = self.open_test
        else:
            raise ValueError("data_range 必须是 'train' 或 'test'")
        
        # 转换为 numpy 数组并对齐长度
        if isinstance(price_data, pd.Series):
            price_arr = price_data.values
            price_index = price_data.index
        else:
            price_arr = np.asarray(price_data)
            price_index = fac_series.index
        
        fac_vals = fac_series.values
        min_len = min(len(fac_vals), len(price_arr))
        fac_vals = fac_vals[:min_len]
        price_arr = price_arr[:min_len]
        
        if hasattr(price_index, '__len__') and len(price_index) >= min_len:
            x_axis = price_index[:min_len]
        else:
            x_axis = np.arange(min_len)
        
        # 清理 NaN 和 Inf
        mask = np.isfinite(fac_vals) & np.isfinite(price_arr)
        if mask.sum() < 10:
            print("有效样本不足，无法绘图。")
            return None
        
        # 创建图形（双 y 轴）
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()
        
        # 绘制价格曲线（左轴）
        color_price = 'tab:blue'
        ax1.set_xlabel('Time / Bar Index', fontsize=11)
        ax1.set_ylabel('Price', color=color_price, fontsize=11)
        line_price = ax1.plot(x_axis, price_arr, color=color_price, linewidth=1.5, 
                              label=f'{price_type.capitalize()} Price', alpha=0.85)
        ax1.tick_params(axis='y', labelcolor=color_price)
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # 绘制因子曲线（右轴）
        color_factor = 'tab:red'
        ax2.set_ylabel(f'{factor_name}', color=color_factor, fontsize=11)
        line_factor = ax2.plot(x_axis, fac_vals, color=color_factor, linewidth=1.2, 
                               label=factor_name, alpha=0.75)
        ax2.tick_params(axis='y', labelcolor=color_factor)
        
        # 添加水平参考线（因子的均值和 ±1σ）
        fac_mean_valid = np.nanmean(fac_vals[mask])
        fac_std_valid = np.nanstd(fac_vals[mask])
        ax2.axhline(fac_mean_valid, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Factor Mean')
        ax2.axhline(fac_mean_valid + fac_std_valid, color='orange', linestyle=':', linewidth=0.8, alpha=0.4, label='±1σ')
        ax2.axhline(fac_mean_valid - fac_std_valid, color='orange', linestyle=':', linewidth=0.8, alpha=0.4)
        
        # 设置标题和图例
        title = f"Factor vs Price: {factor_name} | {data_range} | {price_type}"
        ax1.set_title(title, fontsize=14, fontweight='bold')
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
        
        # 格式化 x 轴（如果是时间索引）
        try:
            if hasattr(x_axis, 'dtype') and 'datetime' in str(x_axis.dtype):
                from matplotlib.dates import DateFormatter, AutoDateLocator
                ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
                ax1.xaxis.set_major_locator(AutoDateLocator())
                fig.autofmt_xdate()
        except Exception:
            pass
        
        plt.tight_layout()
        
        # 保存或显示
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            fname = f"factor_vs_price_{factor_name}_{data_range}.png"
            # 清理文件名
            fname = fname.replace("/", "_").replace("\\", "_").replace(" ", "_").replace("(", "").replace(")", "")
            out_path = save_dir / fname
            out_path.unlink(missing_ok=True)
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"因子与价格对比图已保存至: {out_path}")
        else:
            plt.show()
        
        print("===== 单因子与价格可视化对比结束 =====\n")
        return fig, (ax1, ax2)
    
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

