import os
import pandas as pd
from datetime import datetime
import pytz
from multiprocessing import Pool, cpu_count

# --- 配置 ---
INPUT_DIR = "stock_data"
OUTPUT_DIR = "results"
TIMEZONE = "Asia/Shanghai"  # 上海时区
# 使用 CPU 核心数减 1 进行并行处理，确保系统稳定
NUM_PROCESSES = max(1, cpu_count() - 1) 

# --- 技术指标计算函数 ---
def calculate_macd(df, short_period=12, long_period=26, signal_period=9):
    """计算 MACD 指标 (DIF, DEA, MACD 柱)"""
    # 确保 '收盘' 列是数字类型 (已在 preprocess_stock 中处理)

    # 计算 EMA
    df['EMA_short'] = df['收盘'].ewm(span=short_period, adjust=False).mean()
    df['EMA_long'] = df['收盘'].ewm(span=long_period, adjust=False).mean()

    # 计算 DIF (快线)
    df['DIF'] = df['EMA_short'] - df['EMA_long']

    # 计算 DEA (慢线 / 信号线)
    df['DEA'] = df['DIF'].ewm(span=signal_period, adjust=False).mean()

    # MACD 柱不用于筛选，但可以计算
    # df['MACD'] = (df['DIF'] - df['DEA']) * 2
    return df

def calculate_kdj(df, n=9, m1=3, m2=3):
    """计算 KDJ 指标 (RSV, K, D, J)"""
    # 确保价格列是数字类型 (已在 preprocess_stock 中处理)

    # 1. 计算 RSV
    low_list = df['最低'].rolling(window=n).min()
    high_list = df['最高'].rolling(window=n).max()
    
    # 使用 np.divide 或 pd.Series.div 进行矢量化除法，处理分母为 0 的情况
    range_diff = high_list - low_list
    df['RSV'] = (df['收盘'] - low_list).div(range_diff.replace(0, float('nan'))) * 100
    df['RSV'] = df['RSV'].fillna(100) # 当 H=L 时，通常认为 RSV=100

    # 2. 计算 K (简单移动平均)
    df['K'] = df['RSV'].ewm(com=m1 - 1, adjust=False).mean()

    # 3. 计算 D (K 的移动平均)
    df['D'] = df['K'].ewm(com=m2 - 1, adjust=False).mean()

    # 4. 计算 J
    df['J'] = 3 * df['K'] - 2 * df['D']
    return df

# --- 数据预处理 (在并行任务中执行) ---
def preprocess_stock(df):
    """将关键列转换为数值类型，并按日期排序"""
    
    # 转换为 datetime 对象并排序
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(by='日期').reset_index(drop=True)
    
    # 确保关键价格列是数值类型
    price_cols = ['开盘', '收盘', '最高', '最低']
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 丢弃任何因为非数字数据导致的 NaN 行 (虽然不理想，但确保指标计算稳定)
    df = df.dropna(subset=price_cols)
    
    return df

# --- 核心筛选逻辑 (在并行任务中执行) ---
def process_stock_file(filename):
    """处理单个股票文件，计算指标并检查模式"""
    filepath = os.path.join(INPUT_DIR, filename)
    stock_code = filename.replace('.csv', '')
    
    try:
        df = pd.read_csv(filepath)
        
        if df.empty or len(df) < 30:
            return None # 数据太少或为空，跳过

        df = preprocess_stock(df)
        
        # 重新检查数据长度
        if len(df) < 30:
             return None

        # 计算指标
        df = calculate_macd(df)
        df = calculate_kdj(df)
        
        # --- 检查模式 ---
        # 获取最后两天的数据
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # 条件 1 & 3: MACD 趋势判断 (多头 & 未走坏)
        macd_bullish_trend = (latest['DIF'] > latest['DEA']) and \
                             (latest['DIF'] > 0 and latest['DEA'] > 0)
        
        # MACD 趋势向上 (多头形态下的上升形态)
        macd_rising = (latest['DIF'] > prev['DIF']) and (latest['DEA'] > prev['DEA'])

        # 条件 2: KDJ J 值死叉形态
        # J 值死叉形态：J 值从前一天的较高水平回落，且 J 值小于等于 K 值 (J > K 变为 J <= K，且 J 值小于 90)
        j_dead_cross = (prev['J'] > prev['K']) and (latest['J'] <= latest['K']) and (latest['J'] < 90)
        
        # 结合判断
        if macd_bullish_trend and macd_rising and j_dead_cross:
            return {
                '股票代码': stock_code,
                '日期': latest['日期'].strftime('%Y-%m-%d'),
                '收盘': latest['收盘'],
                'MACD_DIF': latest['DIF'],
                'MACD_DEA': latest['DEA'],
                'KDJ_J': latest['J'],
                'KDJ_K': latest['K'],
                'KDJ_D': latest['D'],
                '说明': 'MACD多头趋势上升，KDJ J值死叉回调，满足右侧加仓条件'
            }
            
    except Exception as e:
        print(f"处理文件 {filename} 时出错: {e}")
        
    return None

# --- 主执行逻辑 ---
def main_optimized():
    stock_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    
    if not os.path.exists(INPUT_DIR) or not stock_files:
        print(f"错误: 找不到 {INPUT_DIR} 目录或其中没有 CSV 文件。")
        return

    print(f"检测到 {len(stock_files)} 支股票，使用 {NUM_PROCESSES} 个进程并行处理...")

    # 使用多进程池并行处理文件
    with Pool(NUM_PROCESSES) as pool:
        # pool.map 会阻塞直到所有结果都返回
        results = pool.map(process_stock_file, stock_files)
        
    # 过滤掉 None 的结果 (即不符合模式的股票)
    all_results = [r for r in results if r is not None]

    # --- 结果输出 ---
    if not all_results:
        print("未发现符合右侧模式的股票。")
        return

    # 转换为 DataFrame
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
    output_filename = f"right_side_pattern_{timestamp_str}.csv"
    output_path = os.path.join(final_output_dir, output_filename)
    
    # 保存结果
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"成功筛选出 {len(all_results)} 条记录，已保存至: {output_path}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if 'GITHUB_ACTIONS' not in os.environ:
        print(f"股票数据期望在 `{INPUT_DIR}/` 目录中。")
    
    main_optimized()
