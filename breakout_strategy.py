import pandas as pd
import sys
import os
from datetime import date, timedelta
import concurrent.futures

# --- 配置参数 ---
DAILY_LIMIT = 0.10 # 涨跌幅限制 (A股主板/中小板 10%)
LIMIT_TOLERANCE = 0.00005 # 涨停容错
STOCK_DATA_DIR = 'stock_data' 
BUY_SIGNAL_FILE = 'buy_signals.csv' 
MAX_WORKERS = 8 

# --- 辅助函数：股票过滤与数据加载 ---

def get_all_stocks_and_filter():
    """
    遍历本地 stock_data 目录下的所有 CSV 文件，
    并排除文件名中包含 'ST'、以 '300' 开头 (创业板) 或 '688' 开头 (科创板) 的股票。
    """
    print(f">>> 正在读取本地目录 '{STOCK_DATA_DIR}' 并过滤股票...")
    
    if not os.path.isdir(STOCK_DATA_DIR):
        print(f"!!! 错误: 找不到股票数据目录 '{STOCK_DATA_DIR}'。请创建并放入CSV文件。")
        return []

    try:
        all_files = os.listdir(STOCK_DATA_DIR)
    except OSError as e:
        print(f"!!! 错误: 无法访问目录 '{STOCK_DATA_DIR}': {e}")
        return []
    
    filtered_list = []
    
    for filename in all_files:
        if not filename.endswith('.csv'):
            continue
            
        ts_code = filename.replace('.csv', '')
        
        # 1. 排除 300 开头的创业板股票
        if ts_code.startswith('300'): 
            print(f"--- 排除创业板股票: {ts_code}")
            continue 
            
        # 2. 排除 688 开头的科创板股票 <-- 关键新增
        if ts_code.startswith('688'):
            print(f"--- 排除科创板股票: {ts_code}")
            continue
            
        # 3. 排除 ST/*ST 股票
        if 'ST' in ts_code.upper(): 
            continue
            
        filtered_list.append(ts_code)
        
    print(f"本地 CSV 文件总数: {len(all_files)}, 过滤后合格股票数: {len(filtered_list)}")
    return filtered_list

def load_stock_data(ts_code):
    """
    从本地加载单个股票的 K 线数据，匹配用户提供的 CSV 格式。
    依赖：'日期', '开盘', '收盘', '最高', '最低', '涨跌幅'
    """
    file_path = os.path.join(STOCK_DATA_DIR, f"{ts_code}.csv")
    if not os.path.exists(file_path):
        return None
        
    try:
        df = pd.read_csv(file_path)
        
        # 1. 重命名列以匹配脚本逻辑
        df.rename(columns={
            '日期': 'trade_date', 
            '开盘': 'open', 
            '收盘': 'close', 
            '最高': 'high', 
            '最低': 'low',
            '涨跌幅': 'change_pct' 
        }, inplace=True)
        
        required_cols = ['trade_date', 'open', 'close', 'high', 'low', 'change_pct']
        if not all(col in df.columns for col in required_cols):
            return None
        
        # 2. 确保日期列是 datetime 对象，并设置为索引
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date').sort_index()
        
        # 3. 手动计算 'pre_close' (前收盘价)
        df['pre_close'] = df['close'].shift(1)
        
        # 4. 将 '涨跌幅' 列从百分比形式转换为小数形式 (例如 10.0 -> 0.10)
        df['change_pct'] = df['change_pct'] / 100.0
        
        # 5. 删除缺失前收盘价的第一行
        df = df.dropna(subset=['pre_close'])
        
        return df
    except Exception as e:
        return None

# --- 辅助函数：策略逻辑判断 ---

def check_consolidation_criteria(df_kline):
    """
    检查涨停后的3天整理K线是否符合条件。
    - 使用 CSV 自带的 'change_pct' 判定涨停。
    - 整理区间只检查 +-3% 范围和不破涨停价。
    """
    if len(df_kline) < 4:
        return False, "数据不足4天"
    
    limit_day = df_kline.iloc[0] # 涨停日 (T-4)
    consolidation_days = df_kline.iloc[1:4] # 整理 3 天 (T-3, T-2, T-1)
    
    # --- 1. 确认第一天是否为涨停 (使用 CSV 自带的涨跌幅列) ---
    limit_price = limit_day['close']
    change_pct = limit_day['change_pct'] 
    min_limit_pct = DAILY_LIMIT - LIMIT_TOLERANCE 
    
    is_limit_hit = change_pct >= min_limit_pct
    
    if not is_limit_hit:
        return False, f"第一天非涨停 (实际涨幅: {change_pct:.4f})"

    # --- 2. 整理不破涨停价 (整理日最低价 >= 涨停价) ---
    if (consolidation_days['low'] < limit_price).any():
        failed_day_index = (consolidation_days['low'] < limit_price).idxmax()
        failed_low = consolidation_days.loc[failed_day_index, 'low']
        return False, f"整理期最低价跌破涨停价: {failed_day_index.strftime('%Y%m%d')} 跌至 {failed_low:.2f} (涨停价 {limit_price:.2f})"

    # --- 3. 每天涨跌幅±3%内 (核心整理要求) ---
    pre_close = limit_price 
    for i in range(len(consolidation_days)):
        row = consolidation_days.iloc[i]
        
        # 检查涨跌幅±3%内 (相对前一交易日收盘价)
        current_change_pct = (row['close'] / pre_close) - 1
        if not (-0.03 <= current_change_pct <= 0.03):
            return False, f"第{i+1}天涨跌幅不在±3%内 (实际涨幅: {current_change_pct:.4f})"
            
        pre_close = row['close']

    return True, "符合买入条件"

# --- 核心函数：线程处理单元 ---

def process_stock(ts_code, trade_date_dt, trade_date_str):
    """
    单个股票的处理逻辑，用于线程池。返回 (信号, 原因)
    """
    df_kline = load_stock_data(ts_code)
    
    if df_kline is None or df_kline.empty:
        return None, f"数据加载失败或为空: {ts_code}"
        
    df_recent = df_kline[df_kline.index <= trade_date_dt]
    recent_4_days = df_recent.tail(4) 
    
    if len(recent_4_days) < 4:
        return None, f"历史数据不足4天: {ts_code}"
        
    is_buy, reason = check_consolidation_criteria(recent_4_days)
    
    if is_buy:
        return {'code': ts_code, 'buy_date': trade_date_str}, "符合买入条件"
        
    return None, f"{ts_code} - 排除原因: {reason}"

# --- 核心函数：策略执行 ---

def run_strategy(trade_date_str):
    """
    执行策略主函数，使用多线程加速。
    """
    print(f"\n--- 策略运行日期 (买入日 T): {trade_date_str} ---")
    
    all_filtered_stocks = get_all_stocks_and_filter()
    if not all_filtered_stocks:
        print("!!! 无股票数据或所有股票已被过滤，策略停止。")
        with open(BUY_SIGNAL_FILE, 'w') as f:
             f.write("code,buy_date\n")
        return

    buy_signals = []
    failure_logs = [] 
    trade_date_dt = pd.to_datetime(trade_date_str, format='%Y%m%d')
    
    print(f"\n>>> 开始使用 {MAX_WORKERS} 线程检查 {len(all_filtered_stocks)} 只股票的整理形态...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_stock = {
            executor.submit(process_stock, ts_code, trade_date_dt, trade_date_str): ts_code 
            for ts_code in all_filtered_stocks
        }
        
        for future in concurrent.futures.as_completed(future_to_stock):
            signal, reason = future.result() 
            if signal:
                buy_signals.append(signal)
            else:
                failure_logs.append(reason)
                
    # 最终输出
    print("\n--- 最终买入列表 (T日执行) ---")
    if buy_signals:
        result_df = pd.DataFrame(buy_signals)
        print(result_df.to_markdown(index=False))
        result_df.to_csv(BUY_SIGNAL_FILE, index=False)
    else:
        print("今日无符合条件的买入信号。")
        with open(BUY_SIGNAL_FILE, 'w') as f:
             f.write("code,buy_date\n")

    # 打印失败日志 (帮助用户调试)
    print("\n--- 股票排除原因日志 (前20条) ---")
    for log in failure_logs[:20]: 
        print(log)
    if len(failure_logs) > 20:
        print(f"... 共 {len(failure_logs)} 个排除日志，仅显示前 20 条。")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_date_str = sys.argv[1]
    else:
        run_date_str = (date.today() - timedelta(days=1)).strftime('%Y%m%d')

    run_strategy(run_date_str)
