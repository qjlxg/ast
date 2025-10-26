import os
import pandas as pd
from datetime import datetime
import pytz
from multiprocessing import Pool, cpu_count
import numpy as np

# --- 配置 ---
INPUT_DIR = "stock_data"
OUTPUT_DIR = "results"
TIMEZONE = "Asia/Shanghai"  # 上海时区
# 使用 CPU 核心数减 1 进行并行处理，提高效率
NUM_PROCESSES = max(1, cpu_count() - 1) 

# --- 技术指标计算函数 ---
def calculate_macd(df, short_period=12, long_period=26, signal_period=9):
    """计算 MACD 指标 (DIF, DEA)"""
    df['EMA_short'] = df['收盘'].ewm(span=short_period, adjust=False).mean()
    df['EMA_long'] = df['收盘'].ewm(span=long_period, adjust=False).mean()
    df['DIF'] = df['EMA_short'] - df['EMA_long']
    df['DEA'] = df['DIF'].ewm(span=signal_period, adjust=False).mean()
    return df

def calculate_kdj(df, n=9, m1=3, m2=3):
    """计算 KDJ 指标 (RSV, K, D, J)"""
    low_list = df['最低'].rolling(window=n).min()
    high_list = df['最高'].rolling(window=n).max()
    
    range_diff = high_list - low_list
    df['RSV'] = (df['收盘'] - low_list).div(range_diff.replace(0, np.nan)) * 100
    df['RSV'] = df['RSV'].fillna(100) 

    df['K'] = df['RSV'].ewm(com=m1 - 1, adjust=False).mean()
    df['D'] = df['K'].ewm(com=m2 - 1, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    return df

# --- 数据预处理 ---
def preprocess_stock(df):
    """将关键列转换为数值类型，并按日期排序"""
    
    # 转换为日期格式并排序
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(by='日期').reset_index(drop=True)
    
    # 确保价格和成交量为数字
    price_volume_cols = ['开盘', '收盘', '最高', '最低', '成交量']
    for col in price_volume_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 丢弃 NaN 行
    df = df.dropna(subset=price_volume_cols)
    
    return df

# --- 核心筛选逻辑 (包含实盘过滤) ---
def process_stock_file(filename):
    """处理单个股票文件，计算指标并检查增强版右侧模式"""
    filepath = os.path.join(INPUT_DIR, filename)
    stock_code = filename.replace('.csv', '')
    
    try:
        df = pd.read_csv(filepath)
        
        df = preprocess_stock(df)
        
        # 1. 数据充足性检查 (至少需要 30 天数据)
        if len(df) < 30:
            return None 

        df = calculate_macd(df)
        df = calculate_kdj(df)
        
        # 滚动计算 20 日收盘价最大值
        df['20D_Max'] = df['收盘'].rolling(window=20).max()
        
        # 获取最后两天的数据
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # 2. 确保前一天的 20D_Max 存在 (即至少有 20 天数据用于计算)
        if df['20D_Max'].iloc[-2] is np.nan:
             return None 
        
        # --- 涨跌停/停牌过滤 (实盘必备) ---
        
        # A. 停牌过滤 (成交量 <= 0)
        if latest['成交量'] <= 0:
            return None
        
        # B. 涨跌停过滤
        if prev['收盘'] <= 0:
            return None
            
        daily_change = (latest['收盘'] / prev['收盘'] - 1)
        
        # 跳过涨停板 (>= 9.5%)
        if daily_change >= 0.095: 
            return None
        
        # 跳过跌停板 (<= -9.5%)
        if daily_change <= -0.095: 
            return None

        # --- 增强版右侧加仓条件 ---
        
        # C. MACD 强多头形态
        macd_bullish = (latest['DIF'] > 0.1) and \
                       (latest['DEA'] > 0.1) and \
                       (latest['DIF'] > latest['DEA'])

        # D. MACD 趋势向上
        macd_rising = (latest['DIF'] > prev['DIF']) and (latest['DEA'] > prev['DEA'])

        # E. KDJ J 值死叉形态
        j_dead_cross = (prev['J'] > prev['K']) and \
                       (latest['J'] <= latest['K']) and \
                       (latest['J'] < 90)

        # F. J 值显著回落 (J值跌幅超过 5%)
        j_drop_percent = 0.0
        j_drop_significant = False
        if prev['J'] > 0:
            j_drop_percent = (prev['J'] - latest['J']) / prev['J']
            j_drop_significant = j_drop_percent > 0.05
        
        # G. 成交量放量 (最新成交量比前一天增加 20% 以上)
        volume_rising = latest['成交量'] > prev['成交量'] * 1.2

        # H. 非新高 (最新收盘价低于前一天计算的 20 日收盘价最高点)
        not_new_high = latest['收盘'] < df['20D_Max'].iloc[-2]

        # 结合所有判断
        if all([macd_bullish, macd_rising, j_dead_cross, j_drop_significant, volume_rising, not_new_high]):
            # --- 构造指定的输出字段 ---
            
            return {
                '股票代码': stock_code,
                '日期': latest['日期'].strftime('%Y-%m-%d'),
                '收盘': round(latest['收盘'], 2),
                '涨跌幅': round(daily_change * 100, 2),
                'J值跌幅%': round(j_drop_percent * 100, 1),
                'MACD_DIF': round(latest['DIF'], 4),
                'MACD_DEA': round(latest['DEA'], 4),
                'KDJ_J': round(latest['J'], 1),
                '说明': '右侧加仓点：MACD多头抬升，KDJ J值显著死叉回调'
            }
            
    except Exception as e:
        # 打印错误信息
        print(f"处理文件 {filename} 时出错: {e}")
        
    return None

# --- 主执行逻辑 (多进程运行) ---
def main_optimized():
    stock_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    
    if not os.path.exists(INPUT_DIR) or not stock_files:
        print(f"错误: 找不到 {INPUT_DIR} 目录或其中没有 CSV 文件。请确保数据已上传。")
        return

    print(f"检测到 {len(stock_files)} 支股票，使用 {NUM_PROCESSES} 个进程并行处理...")

    # 使用多进程池并行处理文件
    with Pool(NUM_PROCESSES) as pool:
        # pool.map 会阻塞直到所有结果都返回
        results = pool.map(process_stock_file, stock_files)
        
    # 过滤掉 None 的结果
    all_results = [r for r in results if r is not None]

    # --- 结果输出到指定目录 ---
    if not all_results:
        print("未发现符合增强版右侧模式的股票。")
        return

    results_df = pd.DataFrame(all_results)
    
    # 获取当前时间（上海时区）
    shanghai_tz = pytz.timezone(TIMEZONE)
    now_shanghai = datetime.now(shanghai_tz)
    
    # 格式化时间戳和路径
    timestamp_str = now_shanghai.strftime("%Y%m%d%H%M%S")
    year_month_dir = now_shanghai.strftime("%Y%m")
    
    # 创建输出目录
    final_output_dir = os.path.join(OUTPUT_DIR, year_month_dir)
    os.makedirs(final_output_dir, exist_ok=True)
    
    # 构造输出文件名
    output_filename = f"right_side_pattern_enhanced_{timestamp_str}.csv"
    output_path = os.path.join(final_output_dir, output_filename)
    
    # 保存结果
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"成功筛选出 {len(all_results)} 条记录，已保存至: {output_path}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main_optimized()
