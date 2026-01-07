import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.utilities.dataframe_functions import impute


if __name__ == "__main__":
    # ==========================================
    # 1. 伪造数据生成 (Mock Data Generation)
    # ==========================================
    # 模拟 24小时 的 5min 数据 (共 288 行)
    periods = 288
    rng = pd.date_range('2024-01-01', periods=periods, freq='5T')

    # 随机游走模拟价格，正态分布模拟其他
    np.random.seed(42)
    df_5min = pd.DataFrame({
        'time': rng,
        'id': 'BTC_USDT', # tsfresh 需要一个 id 列来区分不同实体
        'close': 40000 + np.cumsum(np.random.randn(periods)) * 100, # 价格
        'volume': np.random.randint(100, 1000, size=periods),       # 成交量
        'oi': 10000 + np.cumsum(np.random.randn(periods)) * 50,     # 持仓量 (Open Interest)
        'funding': np.random.normal(0.0001, 0.00005, size=periods)  # 资金费率
    })

    print(f"原始 5min 数据维度: {df_5min.shape}")
    print(df_5min.head())

    # ==========================================
    # 2. 构建滚动窗口 (Rolling Window) - 核心步骤
    # ==========================================
    # 我们想预测 1H 后的行情，所以特征窗口长度为 1小时 (12个 5min bar)
    # roll_time_series 会把数据"展开"，让每一行都带上它过去 N 行的历史数据

    df_rolled = roll_time_series(
        df_5min, 
        column_id='id', 
        column_sort='time',
        max_timeshift=11, # 11代表：包含当前行 + 过去11行 = 总共12行 (1小时)
        min_timeshift=11  # 少于12行的数据（最开始的1小时）不计算特征，防止噪音
    )

    print(f"\n滚动展开后的数据维度 (膨胀约12倍): {df_rolled.shape}")
    # 注意：df_rolled 的 'id' 变成了元组 (原始ID, 当前时间戳)，以此作为特征行的唯一索引

    # ==========================================
    # 3. 定制特征提取参数 (Custom Settings)
    # ==========================================
    # 不要用默认设置，那样会生成几千个特征，慢且容易过拟合。
    # 我们只提取最有物理意义的几类：熵(混乱度)、趋势(斜率)、波峰(阻力)、C3(非线性)

    # 定义通用参数模板
    common_params = {
        # 1. 线性趋势 (判断 OI/价格 方向和力度)
        "linear_trend": [{"attr": "slope"}, {"attr": "stderr"}],
        
        # 2. 近似熵 (判断是震荡还是趋势，数值越小越有序)
        "approximate_entropy": [{"m": 2, "r": 0.2}],
        
        # 3. 复杂性不变距离 (判断曲线是平滑拉升还是锯齿震荡)
        "cid_ce": [{"normalize": True}],
        
        # 4. 超过均值的次数 (判断情绪是否亢奋/拥挤)
        "count_above_mean": None,
        
        # 5. 波峰数量 (判断阻力位密度)
        "number_peaks": [{"n": 1}],
        
        # 6. 非线性自相关 (捕捉反转信号)
        "c3": [{"lag": 1}, {"lag": 2}]
    }

    # 为不同列指定不同的参数 (也可以共用)
    kind_to_fc_parameters = {
        "close": common_params,
        "oi": common_params,      # OI 用这些参数特别有效
        "funding": common_params, # 费率也用这些
        "volume": {               # 成交量我们只关心能量和爆发，不需要看趋势
            "abs_energy": None,
            "kurtosis": None,     # 峰度 (检查是否有巨量柱子)
            "skewness": None
        }
    }

    # ==========================================
    # 4. 特征提取 (Feature Extraction)
    # ==========================================
    print("\n开始提取特征 (这可能需要几秒钟)...")

    X = extract_features(
        df_rolled, 
        column_id='id', 
        column_sort='time',
        kind_to_fc_parameters=kind_to_fc_parameters, # 使用定制参数
        n_jobs=0 # 使用所有 CPU 核心并行计算
    )

    # ==========================================
    # 5. 后处理 (Post-processing)
    # ==========================================
    # tsfresh 可能会产生 NaN (例如方差为0时计算除法)，需要填充
    impute(X)

    # 恢复索引名为时间，方便查看
    X.index = X.index.map(lambda x: x[1]) 
    X.index.name = 'timestamp'

    print("\n------------------------------------------------")
    print(f"特征提取完成！")
    print(f"生成的特征矩阵维度: {X.shape}")
    print("------------------------------------------------")

    print("\n部分提取结果示例:")
    print(X)

    # ==========================================
    # 6. (可选) 保存或直接用于后续模型
    # ==========================================
    # X.to_csv('features_1h.csv')
    # 此时 X 的每一行代表：截止到该时间点，过去1小时内 5min 数据的微观统计特征
    # 你可以直接把 X 喂给 XGBoost / LightGBM 预测下一小时的收益率