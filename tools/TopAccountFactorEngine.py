import polars as pl

class TopAccountEngine:
    def __init__(self, resample_freq='1h'):
        self.freq = resample_freq

    def process(self, raw_account_df: pl.DataFrame, raw_position_df: pl.DataFrame = None):
        """
        :param raw_account_df: 本次提供的【大户账户数】数据
        :param raw_position_df: 之前提供的【大户持仓量】数据 (用于计算背离)
        """
        print(f"[*] 启动大户账户引擎 | 频率: {self.freq}")
        
        # 1. 清洗账户数据
        lf_acc = self._clean_data(raw_account_df)
        
        # 2. 基础特征
        lf_acc = self._build_base_features(lf_acc)
        
        # 3. 核心：如果提供了持仓数据，计算【巨鲸背离】
        if raw_position_df is not None:
            print("[-] 正在计算【巨鲸-普鲸背离】(资金 vs 人数)...")
            lf_pos = self._clean_data(raw_position_df)  # 复用清洗逻辑
            
            # 对齐数据
            # 假设两份数据都是 Top Trader 数据，时间戳应该比较接近
            lf_merged = lf_acc.join(lf_pos, on="open_time", how="inner", suffix="_pos")
            
            # 计算背离
            lf_final = self._build_civil_war_features(lf_merged)
        else:
            lf_final = lf_acc

        return lf_final.collect()

    def _clean_data(self, df: pl.DataFrame):
        """
        通用清洗逻辑：处理 Unix 毫秒时间戳 + 去重
        """
        # 识别时间列名
        time_col = "timestamp" if "timestamp" in df.columns else "open_time"
        
        q = (
            df.lazy()
            .unique(subset=[time_col], keep="last")
        )
        
        # 统一时间格式
        if time_col == "timestamp":
            q = q.with_columns(
                pl.col("timestamp").cast(pl.Int64).cast(pl.Datetime("ms")).alias("open_time")
            )
        else:
            # 如果是字符串 "2025-..."
            q = q.with_columns(
                pl.col("open_time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
            )

        return (
            q.sort("open_time")
            .group_by_dynamic("open_time", every=self.freq)
            .agg([
                # 提取多空占比，忽略 Ratio 字段
                pl.col("longAccount").last().cast(pl.Float64),
                pl.col("shortAccount").last().cast(pl.Float64)
            ])
        )

    def _build_base_features(self, lf: pl.LazyFrame):
        return lf.with_columns([
            # 净账户占比: (Long Accounts - Short Accounts)
            (pl.col("longAccount") - pl.col("shortAccount")).alias("whale_acc_net_ratio")
        ])

    def _build_civil_war_features(self, lf: pl.LazyFrame):
        """
        【核心逻辑】计算 资金(Position) 与 人数(Account) 的背离
        """
        # 预处理：计算 Position 数据的净值 (假设 join 后后缀为 _pos)
        # longAccount_pos 代表持仓量比例
        lf = lf.with_columns([
            (pl.col("longAccount_pos") - pl.col("shortAccount_pos")).alias("whale_money_net_ratio")
        ])
        
        return lf.with_columns([
            # 1. 巨鲸背离因子 (Super Whale Divergence)
            # 逻辑：资金意图 - 人数意图
            # 正值 = 钱比人多 (超级巨鲸在买，普通大户在卖) -> 极度看涨
            # 负值 = 人比钱多 (普通大户在买，超级巨鲸在卖) -> 极度看跌 (诱多)
            (pl.col("whale_money_net_ratio") - pl.col("whale_acc_net_ratio"))
            .alias("feat_super_whale_divergence"),
            
            # 2. 归一化差异 (Z-Score 差值可能更稳健，视数据分布而定)
            # 这里简单做差分演示
        ])