"""
测试脚本：验证closed和label参数的正确设置

运行此脚本来确认您的数据应该使用什么参数
"""

import pandas as pd
import numpy as np
import sys

def test_with_sample_data():
    """使用示例数据测试"""
    print("="*80)
    print("测试1: 使用示例数据（模拟Binance格式）")
    print("="*80)
    
    # 创建测试数据：时间戳=Open Time（模拟Binance）
    timestamps = pd.date_range('2024-01-01 09:00', periods=12, freq='15min')
    df = pd.DataFrame({
        'o': range(100, 112),
        'h': range(101, 113),
        'l': range(99, 111),
        'c': [x + 0.5 for x in range(100, 112)],
        'vol': [1000] * 12,
        'vol_ccy': [10000] * 12,
        'trades': [100] * 12,
    }, index=timestamps)
    
    print("\n原始数据（15分钟）:")
    print(df)
    print(f"\n时间范围: {df.index.min()} ~ {df.index.max()}")
    print(f"数据行数: {len(df)}")
    
    # 方法1: closed='left', label='left'
    print("\n" + "-"*80)
    print("方法1: closed='left', label='left' (适合Open Time)")
    print("-"*80)
    
    result_left = df.resample('1H', closed='left', label='left').agg({
        'o': 'first',
        'h': 'max',
        'l': 'min',
        'c': 'last',
        'vol': 'sum',
        'vol_ccy': 'sum',
        'trades': 'sum',
    })
    
    print("\n结果:")
    print(result_left)
    print(f"\n时间范围: {result_left.index.min()} ~ {result_left.index.max()}")
    print(f"数据行数: {len(result_left)}")
    
    # 方法2: closed='right', label='right'
    print("\n" + "-"*80)
    print("方法2: closed='right', label='right' (适合Close Time)")
    print("-"*80)
    
    result_right = df.resample('1H', closed='right', label='right').agg({
        'o': 'first',
        'h': 'max',
        'l': 'min',
        'c': 'last',
        'vol': 'sum',
        'vol_ccy': 'sum',
        'trades': 'sum',
    })
    
    print("\n结果:")
    print(result_right)
    print(f"\n时间范围: {result_right.index.min()} ~ {result_right.index.max()}")
    print(f"数据行数: {len(result_right)}")
    
    # 对比分析
    print("\n" + "="*80)
    print("对比分析")
    print("="*80)
    
    print(f"\n原始数据: {len(df)} 行，从 {df.index.min()} 到 {df.index.max()}")
    print(f"预期聚合: {len(df) / 4} 行 (15min → 1H, 每4条合并1条)")
    
    print(f"\n方法1结果: {len(result_left)} 行")
    print(f"  - 第一个桶: {result_left.index[0]}")
    print(f"  - 最后一个桶: {result_left.index[-1]}")
    print(f"  - 第一个桶包含的原始数据: {df.loc[df.index[0]:df.index[3]].index.tolist()}")
    
    print(f"\n方法2结果: {len(result_right)} 行")
    print(f"  - 第一个桶: {result_right.index[0]}")
    print(f"  - 最后一个桶: {result_right.index[-1]}")
    
    # 分析差异
    print("\n" + "="*80)
    print("差异分析")
    print("="*80)
    
    if len(result_left) > len(result_right):
        print("⚠️  方法1保留了更多数据")
        print(f"   差异: {len(result_left) - len(result_right)} 行")
        print("   原因: 方法2可能丢失了边界数据")
    elif len(result_right) > len(result_left):
        print("⚠️  方法2保留了更多数据")
        print(f"   差异: {len(result_right) - len(result_left)} 行")
    else:
        print("✅ 两种方法数据量相同")
    
    # 建议
    print("\n" + "="*80)
    print("建议")
    print("="*80)
    
    print("""
对于Binance等交易所数据（时间戳=Open Time）：
  ✅ 推荐使用: closed='left', label='left'
  
  理由：
  1. 时间戳09:00表示[09:00, 09:01)的K线
  2. 聚合时应该从09:00开始，包含[09:00, 09:15, 09:30, 09:45]
  3. 符合金融市场的习惯
    """)


def test_with_real_data(file_path):
    """使用真实数据测试"""
    print("\n" + "="*80)
    print("测试2: 使用真实数据")
    print("="*80)
    
    try:
        # 读取数据
        if file_path.endswith('.gz'):
            df = pd.read_csv(file_path, compression='gzip', index_col=0, parse_dates=True)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        elif file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
        else:
            print(f"❌ 不支持的文件格式: {file_path}")
            return
        
        df = df.sort_index()
        
        print(f"\n✅ 成功读取数据: {file_path}")
        print(f"数据行数: {len(df)}")
        print(f"时间范围: {df.index.min()} ~ {df.index.max()}")
        print(f"时间间隔: {df.index[1] - df.index[0]}")
        
        # 检查列名
        print(f"\n列名: {df.columns.tolist()}")
        
        # 检查时间戳模式
        print("\n时间戳分析:")
        first_5 = df.index[:5]
        for i, ts in enumerate(first_5):
            print(f"  [{i}] {ts.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 判断时间戳类型
        if all(ts.second == 0 for ts in first_5):
            print("\n✅ 时间戳都是整分钟 → 很可能是Open Time格式")
            print("   推荐使用: closed='left', label='left'")
        else:
            print("\n⚠️  时间戳不是整分钟 → 请手动确认")
        
        # 取前100行数据进行测试
        df_sample = df.head(100)
        
        # 推断原始频率
        time_diff = (df.index[1] - df.index[0]).total_seconds() / 60
        original_freq = f"{int(time_diff)}min"
        
        # 选择聚合频率（比原始频率大4倍）
        if time_diff == 15:
            target_freq = '1H'
            expected_ratio = 4
        elif time_diff == 5:
            target_freq = '30min'
            expected_ratio = 6
        elif time_diff == 1:
            target_freq = '15min'
            expected_ratio = 15
        else:
            target_freq = '1H'
            expected_ratio = int(60 / time_diff)
        
        print(f"\n聚合测试: {original_freq} → {target_freq}")
        
        # 测试两种方法
        result_left = df_sample.resample(target_freq, closed='left', label='left').agg({
            'c': 'last' if 'c' in df.columns else 'mean'
        })
        
        result_right = df_sample.resample(target_freq, closed='right', label='right').agg({
            'c': 'last' if 'c' in df.columns else 'mean'
        })
        
        print(f"\n方法1 (closed='left', label='left'): {len(result_left)} 行")
        print(f"方法2 (closed='right', label='right'): {len(result_right)} 行")
        print(f"预期: 约 {len(df_sample) / expected_ratio:.1f} 行")
        
        # 比较哪个更接近预期
        diff_left = abs(len(result_left) - len(df_sample) / expected_ratio)
        diff_right = abs(len(result_right) - len(df_sample) / expected_ratio)
        
        if diff_left < diff_right:
            print("\n✅ 推荐使用: closed='left', label='left'")
        elif diff_right < diff_left:
            print("\n⚠️  您的数据可能适合: closed='right', label='right'")
        else:
            print("\n两种方法结果相同，请根据数据源文档确认")
        
    except Exception as e:
        print(f"\n❌ 读取数据失败: {e}")
        import traceback
        traceback.print_exc()


def check_binance_format():
    """检查Binance数据格式说明"""
    print("\n" + "="*80)
    print("Binance K线数据格式说明")
    print("="*80)
    
    print("""
Binance官方K线数据格式:
[
  1499040000000,      // 0: 开盘时间 (Open time) ← 这是时间戳列
  "0.01634790",       // 1: 开盘价 (Open)
  "0.80000000",       // 2: 最高价 (High)
  "0.01575800",       // 3: 最低价 (Low)
  "0.01577100",       // 4: 收盘价 (Close)
  "148976.11427815",  // 5: 成交量 (Volume)
  1499644799999,      // 6: 收盘时间 (Close time)
  "2434.19055334",    // 7: 成交额
  308,                // 8: 成交笔数
  "1756.87402397",    // 9: 主动买入成交量
  "28.46694368",      // 10: 主动买入成交额
  "17928899.62484339" // 11: 忽略
]

关键点:
1. 第0列（index=0）是 Open time - 开盘时间
2. 这意味着时间戳表示K线的开始时间
3. 例如: timestamp=09:00 表示 [09:00, 09:01) 这一分钟的K线

因此:
  ✅ 应该使用: closed='left', label='left'
  ❌ 不应使用: closed='right', label='right'
    """)


if __name__ == '__main__':
    # 测试1: 示例数据
    test_with_sample_data()
    
    # 测试2: 真实数据（如果提供了文件路径）
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        test_with_real_data(file_path)
    else:
        print("\n" + "="*80)
        print("提示: 可以传入真实数据文件路径进行测试")
        print("="*80)
        print("\n用法:")
        print("  python test_closed_label.py /path/to/your/data.csv")
        print("  python test_closed_label.py /path/to/your/data.csv.gz")
    
    # Binance格式说明
    check_binance_format()
    
    print("\n" + "="*80)
    print("总结建议")
    print("="*80)
    print("""
对于加密货币交易所数据（Binance, OKX等）:
  ✅ 推荐: closed='left', label='left'
  
需要修改的位置:
  1. dataload.py Line 794
  2. dataload.py Line 1089-1090
  
修改方法:
  closed='right' → closed='left'
  label='right'  → label='left'
    """)

