import os
from tardis_dev import datasets, get_exchange_details
import pandas as pd
from tardis_client import TardisClient, Channel
import pandas as pd
import numpy as np
import asyncio
from typing import Optional, List
from tardis_secrets import TARDIS_API_KEY as API_KEY
DOWNLOAD_DIR = "/Users/aming/data/ETHUSDT"

# 常见候选（不同版本/账户权限可能略有差异；跑一下就知道你账户能用哪些）
CANDIDATE_EXCHANGES = [
    # "binance",          # 现货
    "binance-futures",  # 合约（很多账户是这个 id）
    # "binance-usdm",     # 有的账户会拆成 usdm/coinm
    # "binance-coinm",
]
client = TardisClient(api_key=API_KEY)

async def replay_data(exchange: str="binance-futures", from_date: str="2023-01-01", to_date: str="2023-01-02", channel_name: str="topLongShortAccountRatio", symbols: Optional[List[str]] = ["ethusdt"]):
    # 设定要下载的时间段 (Tardis 保存了数年的历史数据)
    messages = client.replay(
        exchange=exchange,
        from_date=from_date,
        to_date=to_date,
        filters=[
            # 获取资金费率 (通常在 ticker 中)
            # Channel(name="derivative_ticker", symbols=["BTCUSDT"]),
            
            # 获取多空数据 (这是 Tardis 的特殊合成频道)
            Channel(name=channel_name, symbols=symbols),
            # Channel(name="topLongShortPositionRatio", symbols=["BTCUSDT"]),
        ]
    )

    # 保存单个月份数据的函数
    def save_month_data(month_str: str, data_list: list, out_dir: str):
        if not data_list:
            return
        df_month = pd.DataFrame(data_list)
        df_month.rename(columns={'time': 'open_time'}, inplace=True)
        df_month['open_time'] = pd.to_datetime(df_month['open_time'], unit='ms')
        df_month.set_index('open_time', inplace=True)
        df_month.sort_index(inplace=True)
        
        out_path = os.path.join(out_dir, f"{channel_name}_{month_str}.csv")
        df_month.to_csv(out_path)
        print(f"✅ 已保存: {month_str} -> {len(df_month)} 条记录")
    
    # 创建输出目录
    out_dir = os.path.join(DOWNLOAD_DIR, channel_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"开始获取数据，保存路径: {out_dir}\n")
    
    # 存储当前月份的数据
    current_month_key = None
    current_month_data = []
    count = 0
    saved_months = 0
    
    # 循环处理每一条历史消息
    async for local_timestamp, message in messages:
        # 处理数据
        msg = message['data']
        count += 1
        
        # 获取时间戳并确定所属月份
        timestamp = pd.to_datetime(msg['time'], unit='ms')
        month_key = timestamp.strftime("%Y-%m")
        
        # 检测到月份切换：保存上个月的数据
        if current_month_key is not None and month_key != current_month_key:
            save_month_data(current_month_key, current_month_data, out_dir)
            saved_months += 1
            current_month_data = []  # 清空，释放内存
        
        # 更新当前月份
        current_month_key = month_key
        current_month_data.append(msg)
        
        # 进度提示
        if count % 3000 == 0:
            print(f"已获取 {count} 条数据 | 当前时间: {timestamp} | 当前月份: {month_key} | 已保存月份: {saved_months}")
    
    # 保存最后一个月的数据
    if current_month_data:
        save_month_data(current_month_key, current_month_data, out_dir)
        saved_months += 1
    
    print(f"\n🎉 数据获取完成！总共 {count} 条数据，保存了 {saved_months} 个月份")


def show_exchange(exchange_id: str):
    try:
        details = get_exchange_details(exchange_id)
    except Exception as e:
        print(f"[{exchange_id}] get_exchange_details failed: {e}")
        return None

    # dataTypes 往往在 details["datasets"]["symbols"][i]["dataTypes"]
    ds = details.get("datasets", {})
    symbols = ds.get("symbols", []) or []
    print(f"\n=== {exchange_id} ===")
    print("symbol count:", len(symbols))
    if symbols:
        # 取前几个 symbol 汇总 dataTypes
        all_types = set()
        for s in symbols[:50]:
            for t in s.get("dataTypes", []) or []:
                all_types.add(t)
        print("sample dataTypes:", sorted(all_types))
        print("sample symbols:", [symbols[i].get("id") for i in range(min(5, len(symbols)))])
    return details

def find_types(details, keywords=("book_snapshot_25", "book_snapshot_5", "book_ticker", "derivative_ticker", "incremental_book_L2", "liquidations", "quotes", "trades")):
    ds = details.get("datasets", {})
    symbols = ds.get("symbols", []) or []
    all_types = set()
    for s in symbols:
        for t in s.get("dataTypes", []) or []:
            all_types.add(t)
    hits = [t for t in sorted(all_types) if any(k.lower() in t.lower() for k in keywords)]
    return hits

def download(exchange_id: str, symbol: str, data_types, from_date: str, to_date: str):
    datasets.download(
        exchange=exchange_id,
        data_types=data_types,
        from_date=from_date,
        format="csv",
        to_date=to_date,     # 非包含
        symbols=[symbol],
        api_key=API_KEY,
        download_dir=DOWNLOAD_DIR,
    )

def list_days(from_date: str, to_date: str, inclusive_end: bool = False) -> Optional[List[str]]:
    """
    获取 from_date 到 to_date 的每一天（YYYY-MM-DD 字符串）。
    - 默认左闭右开：[from_date, to_date)，与 tardis_dev 的 to_date 非包含语义一致
    - inclusive_end=True 时为闭区间：[from_date, to_date]
    """
    # pandas>=1.4 支持 inclusive=...
    inclusive = "both" if inclusive_end else "left"
    return pd.date_range(start=from_date, end=to_date, freq="D", inclusive=inclusive).strftime("%Y-%m-%d").tolist()

def read_liquidations_data(symbol: str="ETHUSDT", from_date: str="2025-02-01", to_date: str="2025-02-02"):
   
    _list_days = list_days(from_date, to_date)
    df_list = []
    for day in _list_days:
        df = pd.read_csv(f"{DOWNLOAD_DIR}/liquidations/binance-futures_liquidations_{day}_{symbol}.csv.gz")
        df.rename(columns={'timestamp': 'open_time'}, inplace=True)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='us')
        df.set_index('open_time', inplace=True)
        df.drop(columns=['symbol', 'local_timestamp', 'id'], inplace=True)
        df_list.append(df)
    result_df = pd.concat(df_list)
    result_df.sort_index(inplace=True)
    return result_df

def deal_liquidations_data(df: pd.DataFrame, freq: str="15min"):
    df['usd_value'] = df['price'] * df['amount']

    # 利用 numpy 的 where 快速拆分多空爆仓
    # side == 'buy' 意味着交易所买入平仓 -> 用户是空头 -> Short Liquidation
    df['short_liq_vol'] = np.where(df['side'] == 'buy', df['usd_value'], 0)
    df['short_liq_cnt'] = np.where(df['side'] == 'buy', 1, 0)

    # side == 'sell' 意味着交易所卖出平仓 -> 用户是多头 -> Long Liquidation
    df['long_liq_vol'] = np.where(df['side'] == 'sell', df['usd_value'], 0)
    df['long_liq_cnt'] = np.where(df['side'] == 'sell', 1, 0)

    # 3. 聚合：Resample 到 15min 级别
    # rule='15min' 表示 15分钟粒度
    # closed='left', label='left' 是量化常用习惯（00:00:00 的数据包含 00:00~00:14:59）
    df_freq = df.resample(freq, label='left', closed='left').agg({
        'short_liq_vol': 'sum',  # 空头爆仓总金额
        'short_liq_cnt': 'sum',  # 空头爆仓次数
        'long_liq_vol':  'sum',  # 多头爆仓总金额
        'long_liq_cnt':  'sum',  # 多头爆仓次数
        'price': ['first', 'max', 'min', 'last'], # (可选) 看看爆仓时的价格分布
        'exchange': 'count'      # 总爆仓单数
    })

    # 4. 扁平化列名 (可选，为了方便访问)
    df_freq.columns = ['_'.join(col).strip() for col in df_freq.columns.values]
    # 修正后列名示例: short_liq_vol_sum, long_liq_vol_sum

    # 5. 缺失值填充
    # 如果某 15min 内没有爆仓，resample 会产生 NaN，需要填 0
    df_freq.fillna(0, inplace=True)

    # 6. 衍生高阶因子 (Feature Engineering)
    # 净爆仓方向：>0 代表空头爆得更多（助涨），<0 代表多头爆得更多（助跌）
    df_freq['net_liq_vol'] = df_freq['short_liq_vol_sum'] - df_freq['long_liq_vol_sum']

    # 爆仓强度比：空头爆仓占比
    df_freq['short_liq_ratio'] = df_freq['short_liq_vol_sum'] / (df_freq['short_liq_vol_sum'] + df_freq['long_liq_vol_sum'] + 1e-6)
    # print(df_15m.head())
    return df_freq

def read_liquidations_data(symbol: str="ETHUSDT", from_date: str="2025-02-01", to_date: str="2025-02-02"):
   
    _list_days = list_days(from_date, to_date)
    df_list = []
    for day in _list_days:
        df = pd.read_csv(f"{DOWNLOAD_DIR}/liquidations/binance-futures_liquidations_{day}_{symbol}.csv.gz")
        df.rename(columns={'timestamp': 'open_time'}, inplace=True)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='us')
        df.set_index('open_time', inplace=True)
        df.drop(columns=['symbol', 'local_timestamp', 'id'], inplace=True)
        df_list.append(df)
    result_df = pd.concat(df_list)
    result_df.sort_index(inplace=True)
    return result_df

def read_derivative_ticker_data(symbol: str="ETHUSDT", from_date: str="2025-02-01", to_date: str="2025-02-02"):
   
    _list_days = list_days(from_date, to_date)
    df_list = []
    for day in _list_days:
        df = pd.read_csv(f"{DOWNLOAD_DIR}/derivative_ticker/binance-futures_derivative_ticker_{day}_{symbol}.csv.gz")
        df.rename(columns={'timestamp': 'open_time'}, inplace=True)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='us')
        df['funding_timestamp'] = pd.to_datetime(df['funding_timestamp'], unit='us')
        df.set_index('open_time', inplace=True)
        df.drop(columns=['exchange', 'local_timestamp', 'symbol'], inplace=True)
        df_list.append(df)
    result_df = pd.concat(df_list)
    result_df.sort_index(inplace=True)
    return result_df

def deal_derivative_ticker_data(df: pd.DataFrame, freq: str="15min"):
    # 2. 聚合到 15min
    # 注意：对于状态数据（OI、费率），我们通常关心该时间窗口结束时的状态
    df_freq = df.resample(freq, label='left', closed='left').agg({
        'funding_rate': 'mean',        # 费率取平均，反映该时段整体情绪
        'predicted_funding_rate': 'last',
        'open_interest': 'last',       # 持仓量取期末值
        'index_price': 'last',
        'mark_price': 'last'
    })

    # 3. 缺失值处理 (Forward Fill)
    # 资金费率和OI不会凭空消失，如果某15分钟没数据，沿用上一时刻的值
    df_freq.fillna(method='ffill', inplace=True)

    # 4. 核心特征构造 (Feature Engineering for gplearn)
    # 这些是真正喂给 gplearn 的因子

    # A. OI 变化率 (OI Momentum)
    # 逻辑：过去 15分钟有多少新仓位进场？
    df_freq['oi_pct_change'] = df_freq['open_interest'].pct_change()

    # B. 期限结构/基差 (Basis)
    # 逻辑：合约比现货贵多少？(反映情绪溢价)
    # 归一化处理：(标记价格 - 指数价格) / 指数价格
    df_freq['basis_ratio'] = (df_freq['mark_price'] - df_freq['index_price']) / df_freq['index_price']

    # C. 费率加速度 (Funding Acceleration)
    # 逻辑：费率是否在快速上升？
    df_freq['funding_rate_delta'] = df_freq['funding_rate'].diff()
    return df_freq

if __name__ == "__main__":
    # for ex in CANDIDATE_EXCHANGES:
    #     details = show_exchange(ex)
    #     if not details:
    #         continue
    #     print("keyword-matched dataTypes:", find_types(details))

    # ---- 你确认好某个 exchange 的 dataTypes 名称后，把下面替换成真实名字即可 ----
    # 例如你找到的可能是 ["open_interest", "liquidations", ...derivative_ticker]（以实际输出为准）
    
    # download("binance-futures", "ETHUSDT", ["derivative_ticker"], "2022-01-01", "2025-11-01")
   
    # df = read_liquidations_data(symbol="ETHUSDT", from_date="2025-02-01", to_date="2025-02-03")
    # df_freq = deal_liquidations_data(df, freq="15min")
    
    
    # df = read_derivative_ticker_data(symbol="ETHUSDT", from_date="2025-02-01", to_date="2025-02-03")
    # print(df.columns)
    # print(df.head())
    # df_freq = deal_derivative_ticker_data(df, freq="15min")
    # print(df_freq.head())


    # print(df_freq.head())


    # https://docs.tardis.dev/historical-data-details/binance-futures topLongShortPositionRatio topLongShortAccountRatio takerlongshortRatio 
    
    asyncio.run(replay_data(exchange="binance-futures", from_date="2022-01-01", to_date="2024-01-01", channel_name="liquidations", symbols=["ethusdt"]))