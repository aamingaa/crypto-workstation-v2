from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math
import matplotlib.pyplot as plt
from tools.LiquidationFactorEngine import LiquidationFactorEngine as liq_factor_engine

def generate_etime_close_data_divd_time(bgn_date, end_date, index_code, frequency):
    """生成行情数据"""
    read_file_path = '/Users/aming/project/python/课程代码合集/单因子评价方式_2/' + index_code + '_' + frequency + '.xlsx'
    kbars = pd.read_excel(read_file_path)
    
    kbars['tdate'] = pd.to_datetime(kbars['etime']).dt.date
    dt = pd.to_datetime(kbars['etime'], format='%Y-%m-%d %H:%M:%S.%f')
    kbars['etime'] = pd.Series([pd.Timestamp(x).round('s').to_pydatetime() for x in dt])
    
    # 根据日期范围筛选数据
    mask = (pd.to_datetime(kbars['etime']) >= pd.to_datetime(bgn_date)) & \
           (pd.to_datetime(kbars['etime']) <= pd.to_datetime(end_date))
    kbars = kbars[mask].reset_index(drop=True)
    
    return kbars[['etime', 'tdate', 'close']].reset_index(drop=True)

def calculate_positions(predictions, scale=0.0005, position_size=1):
    """计算持仓信号 sigmoid  tanh, relu, zscore, """
    positions = (predictions / scale) * position_size
    return np.clip(positions, -1, 1)

# 时间轴:     t0 ————→ t1 ————→ t2 ————→ t3 ————→ t4
# 持仓:       0.5      0.8     -0.2      0.3      0.1
# 价格变化:      ↑2%      ↓1%      ↑2%      ↑2%
# 策略收益:            1%      -0.8%    -0.4%     0.6%

# 对应关系:
# t0持仓(0.5) × t0→t1收益(2%) = t1策略收益(1%)
# t1持仓(0.8) × t1→t2收益(-1%) = t2策略收益(-0.8%)
# t2持仓(-0.2) × t2→t3收益(2%) = t3策略收益(-0.4%)
# t3持仓(0.3) × t3→t4收益(2%) = t4策略收益(0.6%)


# t0时刻：
# - 收到信号：下一期应该持仓0.5
# - 执行交易：买入0.5的仓位
# - 持仓状态：0.5

# t0→t1期间：
# - 持仓0.5不变
# - 价格变化2%
# - 策略收益：0.5 × 2% = 1%

# t1时刻：
# - 收到新信号：下一期应该持仓0.8  
# - 执行交易：从0.5调整到0.8（增加0.3）
# - 持仓状态：0.8

# t1→t2期间：
# - 持仓0.8不变
# - 价格变化-1%
# - 策略收益：0.8 × (-1%) = -0.8%

def calculate_returns_and_nav(data, positions):
    """计算收益率和净值"""
    price_changes = data['close'].pct_change().fillna(0)
    strategy_returns = positions[:-1] * price_changes[1:]
    # position_diff = np.abs(positions[1:] - positions[:-1]) 80% -10%  90%
    # pnl = strategy_returns
    # net_pnl = pnl - fees*position_diff
    # cumu_cost = np.cumsum(fees*position_diff)
    cumulative_returns = (1 + strategy_returns).cumprod() # 注意这里，可以修正为单利模式，和加入手续费
    # cumulative_returns = 1 + strategy_returns.cumsum() # 注意这里，可以修正为单利模式，和加入手续费
    return strategy_returns, cumulative_returns

def calculate_metrics_by_period(data, positions, returns, nav):
   """计算给定时期的评价指标"""
   metrics = {}
   rf = 0.03
    # param_metric = weights2calculate([0.5,0.2, 0.3])
   
   # 基础指标计算
   annual_return = (nav.iloc[-1] ** (252/len(nav))) - 1
   annual_vol = returns.std() * np.sqrt(252)
   sharpe = (annual_return - rf) / annual_vol if annual_vol != 0 else 0
   

   # 计算最大回撤及其起始日期
#    回撤 = 历史最高净值 - 当前净值
   drawdown = nav.cummax() - nav
#    最大回撤率 = 最大回撤金额 / 历史最高净值
   max_drawdown = drawdown.max() / nav.cummax().max()
   max_drawdown_end_idx = drawdown.argmax()
   max_drawdown_start_idx = nav[:max_drawdown_end_idx+1].argmax()
   
   max_drawdown_start_date = data['tdate'].iloc[max_drawdown_start_idx]
   max_drawdown_end_date = data['tdate'].iloc[max_drawdown_end_idx]
   
   # 计算胜率和盈亏比
   daily_pnl = returns * positions[:-1]
   win_rate = (daily_pnl > 0).mean()

   # 盈亏比 = 盈利的平均值 / 亏损的平均值
   gain_loss_ratio = abs(daily_pnl[daily_pnl > 0].mean() / daily_pnl[daily_pnl < 0].mean()) \
                    if len(daily_pnl[daily_pnl < 0]) > 0 else np.inf
#    防止除零错误：如果没有亏损交易日，返回无穷大
   # 衡量因子的择时有效性，或者因子的稳定性，当前bar对应的，过去100个bar，他的gain_loss_ratio，

   metrics.update({
       '总收益': nav.iloc[-1] - 1,
       '年化收益': annual_return,
       '年化波动率': annual_vol,
       '夏普比率': sharpe,
       '最大回撤率': max_drawdown,
       '最大回撤起始日': max_drawdown_start_date,
       '最大回撤结束日': max_drawdown_end_date,
       '总交易次数': len(returns),
       '胜率': win_rate,
       '盈亏比': gain_loss_ratio
   })
   
   return metrics



def backtest(original_data, index_code, frequency, n_days):
    """回测主函数"""
    # 数据预处理
    final_frame = original_data[['tdate', 'etime', 'close', 'fct']].dropna(axis=0).reset_index(drop=True)
    # 如果应对多个品种的计算，可以用下面的方式。
    # raw_data, fct_data = {},{}
    # raw_data['btc_30m'] = {} 
    # fct_data['btc_30m'] = {} 
    
    # 计算时间步长和收益率
    t_delta = int(1 * n_days) if frequency == '15' else int(int(240 / int(frequency)) * n_days)
    final_frame['ret'] = final_frame['close'].shift(-t_delta) / final_frame['close'] - 1 # 这里的return，是我们的label，是未来t_delta天的收益率，他和我们计算绩效的时候，不一样。
    final_frame = final_frame.dropna(axis=0).reset_index(drop=True)

    # 训练测试集划分
    train_set_end_index = final_frame[(final_frame['etime'].dt.year == 2019) & 
                                    (final_frame['etime'].dt.month == 12) & 
                                    (final_frame['etime'].dt.day == 31)].index[0]
    
    # 准备训练数据
    X_train = final_frame.loc[:train_set_end_index, 'fct'].values.reshape(-1, 1)
    y_train = final_frame.loc[:train_set_end_index, 'ret'].values.reshape(-1, 1)
    X_test = final_frame.loc[train_set_end_index+1:, 'fct'].values.reshape(-1, 1)

    # 模型训练和预测 Y = AX + B
    model = LinearRegression(fit_intercept=True) # t-statistic，weight，bias
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train).flatten()
    y_test_pred = model.predict(X_test).flatten()

    # 初始化结果DataFrame
    indicators_frame = pd.DataFrame()
    
    # 计算每年的指标
    for year in final_frame['etime'].dt.year.unique():
        year_mask = final_frame['etime'].dt.year == year
        year_data = final_frame[year_mask]
        
        if len(year_data) > 0:
            year_positions = calculate_positions(
                model.predict(year_data['fct'].values.reshape(-1, 1)).flatten()
            )
            year_returns, year_nav = calculate_returns_and_nav(year_data, year_positions)
            year_metrics = calculate_metrics_by_period(
                year_data, year_positions, year_returns, year_nav
            )
            indicators_frame = pd.concat([
                indicators_frame, 
                pd.DataFrame(year_metrics, index=[year])
            ])
    
    # 计算样本内外指标
    train_data = final_frame[:train_set_end_index+1]
    test_data = final_frame[train_set_end_index+1:]
    
    # 样本内指标
    train_positions = calculate_positions(y_train_pred)
    train_returns, train_nav = calculate_returns_and_nav(train_data, train_positions)
    indicators_frame.loc['样本内'] = calculate_metrics_by_period(
        train_data, train_positions, train_returns, train_nav
    )
    
    # 样本外指标
    test_positions = calculate_positions(y_test_pred)
    test_returns, test_nav = calculate_returns_and_nav(test_data, test_positions)
    indicators_frame.loc['样本外'] = calculate_metrics_by_period(
        test_data, test_positions, test_returns, test_nav
    )
    
    # 总体指标
    total_positions = np.concatenate([train_positions, test_positions])
    total_returns, total_nav = calculate_returns_and_nav(final_frame, total_positions)
    indicators_frame.loc['总体'] = calculate_metrics_by_period(
        final_frame, total_positions, total_returns, total_nav
    )
    
#     年度计算：时间维度的策略分析
# 样本内外计算：模型质量的技术分析

    return indicators_frame


# ============================================================
# 下面是“简单版”的批量因子评估（更适合爆仓/分层挖掘出来的一堆因子）
# - 不改动上面的 backtest()，避免影响你原来网格搜索脚本
# - 直接复用：线性回归(样本内拟合) -> 预测 -> 仓位 -> 回测指标
# - 额外提供：IC / RankIC、手续费、换手率、Top/Bottom 分位多空
# ============================================================

def _infer_annual_bars_from_freq(freq: str) -> int:
    """
    简单推断年化 bar 数：
    - '15' / '15m' -> 365*24*4 = 35040（加密货币 7*24）
    - 其它 -> 252（偏股票日频习惯）
    """
    if freq is None:
        return 252
    f = str(freq).lower().strip()
    if f in ("15", "15m", "15min", "15mins", "15minute", "15minutes"):
        return 365 * 24 * 4
    return 252


def _calc_ic_and_rankic(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    返回 (IC, RankIC)
    - IC: Pearson corr
    - RankIC: 对 x/y 做 rank 后再算 Pearson corr（避免依赖 scipy）
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 50:
        return (np.nan, np.nan)
    xv = x[mask]
    yv = y[mask]
    try:
        ic = float(np.corrcoef(xv, yv)[0, 1])
    except Exception:
        ic = np.nan
    try:
        xr = pd.Series(xv).rank(method="average").values
        yr = pd.Series(yv).rank(method="average").values
        rankic = float(np.corrcoef(xr, yr)[0, 1])
    except Exception:
        rankic = np.nan
    return (ic, rankic)


def _build_quantile_long_short_pos(factor_vals: np.ndarray, n_quantiles: int = 5) -> np.ndarray:
    """顶分位 +1，底分位 -1，其它 0（简单稳定，适合单因子 sanity check）"""
    s = pd.Series(np.asarray(factor_vals, dtype=float))
    if s.nunique() <= 1:
        return np.zeros(len(s), dtype=float)
    try:
        q = pd.qcut(s.rank(method="first"), q=n_quantiles, labels=False, duplicates="drop")
    except Exception:
        return np.zeros(len(s), dtype=float)
    pos = np.zeros(len(s), dtype=float)
    if q.max() == q.min():
        return pos
    pos[q == q.max()] = 1.0
    pos[q == q.min()] = -1.0
    return pos


def _simulate_pnl_from_pos(
    close: np.ndarray,
    pos: np.ndarray,
    fees_rate: float = 0.0005,
    annual_bars: int = 35040,
) -> tuple[np.ndarray, dict]:
    """
    用最简交易假设做回测：
    - bar 收益用 close-to-close 的 log return
    - 手续费按仓位变动的绝对值扣：abs(diff(pos)) * fees_rate
    返回 (pnl_cumsum_log, metrics)
    """
    close = np.asarray(close, dtype=float).reshape(-1)
    pos = np.asarray(pos, dtype=float).reshape(-1)
    n = min(len(close), len(pos))
    close = close[:n]
    pos = pos[:n]
    if n < 5:
        return np.zeros(n, dtype=float), {
            "Annual Return": 0.0,
            "Sharpe Ratio": 0.0,
            "MAX_Drawdown": 0.0,
            "Turnover_per_bar": 0.0,
        }

    rets = np.concatenate([[0.0], np.diff(np.log(close))])  # 每bar log return
    pos_change = np.concatenate([[0.0], np.diff(pos)])
    gain_loss = pos * rets - np.abs(pos_change) * float(fees_rate)
    pnl = np.cumsum(gain_loss)

    mean_ret = float(np.mean(gain_loss))
    std_ret = float(np.std(gain_loss)) if len(gain_loss) > 1 else 0.0
    annual_ret = mean_ret * int(annual_bars)
    sharpe = annual_ret / (std_ret * math.sqrt(int(annual_bars))) if std_ret > 0 else 0.0

    equity = np.exp(pnl)
    peak = np.maximum.accumulate(equity)
    peak = np.where(peak == 0, 1.0, peak)
    drawdown = equity / peak - 1.0
    max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

    turnover = float(np.mean(np.abs(pos_change))) if len(pos_change) > 0 else 0.0

    metrics = {
        "Annual Return": annual_ret,
        "Sharpe Ratio": float(sharpe),
        "MAX_Drawdown": max_dd,
        "Turnover_per_bar": turnover,
    }
    return pnl, metrics


def evaluate_factors_simple(
    price_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    freq: str = "15m",
    horizon_bars: int = 8,
    train_end_time=None,
    split_ratio: float = 0.7,
    n_quantiles: int = 5,
    fees_rate: float = 0.0005,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    批量评估 factors_df 的每一列因子（简单版，不搞太多抽象）：
    1) 对齐时间
    2) 构造未来收益 label（horizon_bars）
    3) 计算 train/test 的 IC、RankIC
    4) 两套回测口径（都很简单）：
       - 回归回测：训练集 fit LinearRegression(factor -> future_ret)，预测后转仓位（复用 calculate_positions）
       - 分位多空：顶分位+1、底分位-1
    输出一个汇总表，默认按 test_sharpe_lr 排序。

    Args:
        price_df: 必须含 'close'，index 或列里有时间（建议 index=DatetimeIndex）
        factors_df: index=DatetimeIndex（建议是 open_time），列为因子
        freq: 仅用于推断年化 bar 数
        horizon_bars: 未来多少期收益作为 label（15m 下 8=2h）
        train_end_time: 指定训练集截止时间（Datetime/str），否则用 split_ratio 切分
    """
    # if price_df is None or factors_df is None:
    #     raise ValueError("price_df / factors_df 不能为空")
    # if "close" not in price_df.columns:
    #     raise ValueError("price_df 必须包含 'close' 列")
    # if factors_df.shape[1] == 0:
    #     raise ValueError("factors_df 没有因子列")

    # 统一索引为 DatetimeIndex
    p = price_df.copy()
    f = factors_df.copy()
    if not isinstance(p.index, pd.DatetimeIndex):
        # 尝试自动找时间列
        for c in ["open_time", "etime", "time", "datetime", "date"]:
            if c in p.columns:
                p[c] = pd.to_datetime(p[c], errors="coerce")
                p = p.set_index(c)
                break
    if not isinstance(f.index, pd.DatetimeIndex):
        for c in ["open_time", "etime", "time", "datetime", "date"]:
            if c in f.columns:
                f[c] = pd.to_datetime(f[c], errors="coerce")
                f = f.set_index(c)
                break
    if not isinstance(p.index, pd.DatetimeIndex) or not isinstance(f.index, pd.DatetimeIndex):
        raise ValueError("price_df / factors_df 需要能对齐的时间索引（DatetimeIndex 或包含可解析的时间列）")

    p = p.sort_index()
    f = f.sort_index()

    # 对齐并构造 label（未来收益）
    df = p[["close"]].join(f, how="inner")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
    df["ret_fwd"] = df["close"].shift(-int(horizon_bars)) / df["close"] - 1.0
    df = df.dropna(subset=["ret_fwd"])

    if len(df) < 200:
        raise ValueError(f"对齐后样本太少：{len(df)}，请检查时间对齐或数据区间")

    # 切分 train/test
    if train_end_time is not None:
        t_end = pd.to_datetime(train_end_time)
        train_mask = df.index <= t_end
        if train_mask.sum() < 50 or (~train_mask).sum() < 50:
            raise ValueError("按 train_end_time 切分后 train/test 样本不足")
        train_df = df[train_mask]
        test_df = df[~train_mask]
    else:
        n_train = int(len(df) * float(split_ratio))
        n_train = max(50, min(n_train, len(df) - 50))
        train_df = df.iloc[:n_train]
        test_df = df.iloc[n_train:]

    annual_bars = _infer_annual_bars_from_freq(freq)

    records = []
    y_train = train_df["ret_fwd"].values.reshape(-1, 1)
    y_test = test_df["ret_fwd"].values.reshape(-1, 1)

    for col in f.columns:
        x_train = train_df[col].values.reshape(-1, 1)
        x_test = test_df[col].values.reshape(-1, 1)

        # IC / RankIC（train/test 分开算）
        ic_tr, ric_tr = _calc_ic_and_rankic(train_df[col].values, train_df["ret_fwd"].values)
        ic_te, ric_te = _calc_ic_and_rankic(test_df[col].values, test_df["ret_fwd"].values)

        # 口径 A：回归 -> 连续仓位（复用原来的 calculate_positions）
        try:
            model = LinearRegression(fit_intercept=True)
            model.fit(x_train, y_train)
            pred_train = model.predict(x_train).flatten()
            pred_test = model.predict(x_test).flatten()

            pos_train = calculate_positions(pred_train)
            pos_test = calculate_positions(pred_test)
            _, m_tr = _simulate_pnl_from_pos(
                close=train_df["close"].values, pos=pos_train, fees_rate=fees_rate, annual_bars=annual_bars
            )
            _, m_te = _simulate_pnl_from_pos(
                close=test_df["close"].values, pos=pos_test, fees_rate=fees_rate, annual_bars=annual_bars
            )
        except Exception:
            m_tr = {"Annual Return": np.nan, "Sharpe Ratio": np.nan, "MAX_Drawdown": np.nan, "Turnover_per_bar": np.nan}
            m_te = {"Annual Return": np.nan, "Sharpe Ratio": np.nan, "MAX_Drawdown": np.nan, "Turnover_per_bar": np.nan}

        # 口径 B：分位多空（更贴近“分层挖掘”的直觉）
        try:
            pos_q_train = _build_quantile_long_short_pos(train_df[col].values, n_quantiles=n_quantiles)
            pos_q_test = _build_quantile_long_short_pos(test_df[col].values, n_quantiles=n_quantiles)
            _, mq_tr = _simulate_pnl_from_pos(
                close=train_df["close"].values, pos=pos_q_train, fees_rate=fees_rate, annual_bars=annual_bars
            )
            _, mq_te = _simulate_pnl_from_pos(
                close=test_df["close"].values, pos=pos_q_test, fees_rate=fees_rate, annual_bars=annual_bars
            )
        except Exception:
            mq_tr = {"Annual Return": np.nan, "Sharpe Ratio": np.nan, "MAX_Drawdown": np.nan, "Turnover_per_bar": np.nan}
            mq_te = {"Annual Return": np.nan, "Sharpe Ratio": np.nan, "MAX_Drawdown": np.nan, "Turnover_per_bar": np.nan}

        records.append({
            "factor": col,
            "IC_train": ic_tr,
            "RankIC_train": ric_tr,
            "IC_test": ic_te,
            "RankIC_test": ric_te,
            "test_sharpe_lr": m_te["Sharpe Ratio"],
            "test_mdd_lr": m_te["MAX_Drawdown"],
            "test_turnover_lr": m_te["Turnover_per_bar"],
            "test_sharpe_q": mq_te["Sharpe Ratio"],
            "test_mdd_q": mq_te["MAX_Drawdown"],
            "test_turnover_q": mq_te["Turnover_per_bar"],
        })

    out = pd.DataFrame(records)
    # 排序优先级：测试集回归回测夏普，其次 |IC|
    out["absIC_test"] = out["IC_test"].abs()
    out = out.sort_values(["test_sharpe_lr", "absIC_test"], ascending=[False, False])

    if top_n is not None and int(top_n) > 0:
        return out.head(int(top_n)).reset_index(drop=True)
    return out.reset_index(drop=True)


def load_factors_csv_simple(path: str, time_col: str = "open_time") -> pd.DataFrame:
    """
    简单读因子 csv：默认假设有 open_time 列；若没有则尝试用第一列做索引。
    """
    df = pd.read_csv(path)
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.set_index(time_col)
    else:
        # 兜底：第一列当时间
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        df = df.set_index(df.columns[0])
    return df.sort_index()


def plot_factor_corr_heatmap(
    factors_df: pd.DataFrame,
    method: str = "spearman",
    max_factors: int = 40,
    figsize: tuple = (10, 8),
    title: str | None = None,
    save_path: str | None = None,
):
    """
    因子相关性 heatmap（最常用，用来查冗余/同义因子）
    - factors_df: index=时间, columns=因子
    - method: 'spearman'（推荐，稳一些）或 'pearson'
    - max_factors: 因子太多会画不清，默认只画前 N 列
    """
    if factors_df is None or factors_df.empty:
        raise ValueError("factors_df 为空，无法画 heatmap")
    df = factors_df.copy()
    # 只保留数值列
    df = df.select_dtypes(include=[np.number])
    if df.shape[1] == 0:
        raise ValueError("factors_df 没有数值因子列")

    cols = list(df.columns)[: int(max_factors)]
    df = df[cols].replace([np.inf, -np.inf], np.nan).dropna(how="all")
    corr = df.corr(method=method)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title(title or f"Factor Correlation Heatmap ({method})")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticklabels(cols, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return corr


def plot_factor_value_heatmap(
    factors_df: pd.DataFrame,
    max_rows: int = 500,
    max_factors: int = 40,
    normalize: str = "zscore",
    figsize: tuple = (12, 6),
    title: str | None = None,
    save_path: str | None = None,
):
    """
    因子值 heatmap（time × factor）
    - normalize='zscore'：按列做 z-score，便于把不同量纲的因子放一张图看
    - 数据太大时默认抽样前 max_rows 行（你也可以先 df.iloc[-max_rows:] 看最近）
    """
    if factors_df is None or factors_df.empty:
        raise ValueError("factors_df 为空，无法画 heatmap")
    df = factors_df.copy()
    df = df.select_dtypes(include=[np.number])
    if df.shape[1] == 0:
        raise ValueError("factors_df 没有数值因子列")

    cols = list(df.columns)[: int(max_factors)]
    df = df[cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how="all")

    # 截断行数，避免图太大
    df = df.iloc[: int(max_rows)]

    if normalize == "zscore":
        mu = df.mean(axis=0)
        sigma = df.std(axis=0).replace(0, np.nan)
        df = (df - mu) / sigma
        df = df.fillna(0.0).clip(-6, 6)
        cmap = "RdBu_r"
        vmin, vmax = -3, 3
    else:
        cmap = "viridis"
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(df.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title or f"Factor Value Heatmap ({normalize})")
    ax.set_xlabel("Factors")
    ax.set_ylabel("Time (rows)")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    # y轴太密就不标
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return df


if __name__ == "__main__":
    # 最小示例（按你的需求：简单直观，不做复杂抽象）
    #
    # 1) 准备 price_df：至少要有 close
    #    - 如果你已经有 ETH 的 15m K 线 DataFrame，确保 index 是时间即可
    # 2) 准备 factors_df：index=时间，列=因子
    #    - 例如 LiquidationFactorEngine().process(...) 的返回值
    #
    # 注意：下面只是示例入口，不会强依赖你的具体数据路径。
    print("[demo] evaluate_factors_simple: 请替换为你的 price_df / factors_df")

    # 示例：从 CSV 读取因子（你也可以直接传 DataFrame）
    # factors_df = load_factors_csv_simple("/path/to/your_factors.csv", time_col="open_time")

    # 示例：构造一个假的 price_df（请替换）
    price_df = pd.read_csv("/path/to/your_ohlcv.csv")
    price_df["open_time"] = pd.to_datetime(price_df["open_time"])
    price_df = price_df.set_index("open_time").sort_index()

    # 你在文件顶部用了：
    # from tools.LiquidationFactorEngine import LiquidationFactorEngine as liq_factor_engine
    # 这里 liq_factor_engine 实际上就是“类”，直接实例化即可
    liq_engine = liq_factor_engine(resample_freq="15m")

    bucket_quantiles = [0.75, 0.90]
    bucket_window_hours=[24, 48]
    mining_windows=[24]
    mining_quantiles=[0.90]

    # liq_factor_df = liq_engine.process(
    #     liq_df,
    #     bucket_quantiles=bucket_quantiles,
    #     bucket_window_hours=bucket_window_hours,
    #     mining_windows=mining_windows,
    #     mining_quantiles=mining_quantiles,
    # )

    # print(liq_factor_df.head())
        
    # 然后跑评估：
    # report = evaluate_factors_simple(
    #     price_df=price_df,
    #     factors_df=factors_df,
    #     freq="15m",
    #     horizon_bars=8,          # 15m*8 = 2h
    #     train_end_time="2025-01-01",
    #     n_quantiles=5,
    #     fees_rate=0.0005,
    #     top_n=30,
    # )
    # print(report)

    # 画 heatmap（把 factors_df 换成你的 liq_factor_df）
    # plot_factor_corr_heatmap(liq_factor_df, method="spearman", max_factors=40)
    # plot_factor_value_heatmap(liq_factor_df, max_rows=500, max_factors=40, normalize="zscore")
