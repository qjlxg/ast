import pandas as pd
import sys
import os
from datetime import date, timedelta

# --- 配置参数 ---
DAILY_LIMIT = 0.10 # 涨跌幅限制 (A股主板/中小板 10%)
STOCK_DATA_DIR = 'stock_data' # 本地股票数据存放目录
BUY_SIGNAL_FILE = 'buy_signals.csv' # 信号输出文件

# --- 辅助函数：股票过滤与数据加载 ---

def get_all_stocks_and_filter():
    """
    遍历本地 stock_data 目录下的所有 CSV 文件，
    并排除文件名中包含 'ST' 或以 '300' 开头的股票。
    """
    print(f">>> 正在读取本地目录 '{STOCK_DATA_DIR}' 并过滤股票...")
    
    if not os.path.isdir(STOCK_DATA_DIR):
        print(f"!!! 错误: 找不到股票数据目录 '{STOCK_DATA_DIR}'。请创建并放入CSV文件。")
        return []

    all_files = os.listdir(STOCK_DATA_DIR)
    filtered_list = []
    
    for filename in all_files:
        if not filename.endswith('.csv'):
            continue
            
        # 股票代码是文件名中排除后缀的部分 (例如 '603301.csv' -> '603301')
        ts_code = filename.replace('.csv', '')
        
        # 1. 排除 300 开头的创业板股票
        if ts_code.startswith('300'): 
            print(f"--- 排除创业板股票: {ts_code}")
            continue 
            
        # 2. 排除 ST/*ST 股票 (基于文件名)
        if 'ST' in ts_code.upper(): 
            print(f"--- 排除 ST 股票: {ts_code}")
            continue
            
        filtered_list.append(ts_code)
        
    print(f"本地 CSV 文件总数: {len(all_files)}, 过滤后合格股票数: {len(filtered_list)}")
    return filtered_list

def load_stock_data(ts_code):
    """
    从本地加载单个股票的 K 线数据，匹配用户提供的 CSV 格式。
    原始列名: 日期, 股票代码, 开盘, 收盘, 最高, 最低, ...
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
            '最低': 'low'
        }, inplace=True)
        
        # 2. 确保关键列存在
        required_cols = ['trade_date', 'open', 'close', 'high', 'low']
        if not all(col in df.columns for col in required_cols):
            print(f"!!! {ts_code} 数据缺失关键列，跳过。")
            return None
        
        # 3. 确保日期列是 datetime 对象，并设置为索引
        # 假设日期格式是 YYYY-MM-DD
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date').sort_index()
        
        # 4. 手动计算 'pre_close' (前收盘价)
        df['pre_close'] = df['close'].shift(1)
        
        # 5. 删除缺失前收盘价的第一行
        df = df.dropna(subset=['pre_close'])
        
        return df
    except Exception as e:
        print(f"!!! 无法加载或处理 {ts_code} 的数据: {e}")
        return None

# --- 辅助函数：策略逻辑判断 (与之前版本相同) ---

def is_small_positive_line(row, pre_close):
    """
    判断是否为“小阳线”且涨跌幅在 [0.5%, 3%] 内。
    """
    change_pct = (row['close'] / pre_close - 1)
    
    return (row['close'] > row['open'] and 
            0.005 <= change_pct <= 0.03)

def check_consolidation_criteria(df_kline):
    """
    检查涨停后的3天整理K线是否符合条件。
    df_kline 应该是包含涨停日和随后3个交易日的K线数据（共4行，涨停日是第一天）。
    """
    if len(df_kline) < 4:
        return False, "数据不足4天"
    
    limit_day = df_kline.iloc[0] # 涨停日 (T-4)
    consolidation_days = df_kline.iloc[1:4] # 整理 3 天 (T-3, T-2, T-1)
    
    # --- 1. 确认第一天是否为涨停 ---
    limit_price = limit_day['close']
    # 使用前收盘价计算涨幅
    is_limit_hit = (limit_day['close'] / limit_day['pre_close'] - 1) >= DAILY_LIMIT - 0.0001
    if not is_limit_hit:
        return False, "第一天非涨停"

    # --- 2. 整理不破涨停价 (整理日最低价 >= 涨停价) ---
    if (consolidation_days['low'] < limit_price).any():
        return False, "整理期最低价跌破涨停价"

    # --- 3. 每天涨跌幅±3%内优选 & 4. 3天均为小阳线优选 ---
    pre_close = limit_price # 整理第一天 (T-3) 的前收盘价
    for i in range(len(consolidation_days)):
        row = consolidation_days.iloc[i]
        
        # 3. 检查涨跌幅±3%内 (相对前一交易日收盘价)
        current_change_pct = (row['close'] / pre_close) - 1
        if not (-0.03 <= current_change_pct <= 0.03):
            return False, f"第{i+1}天涨跌幅不在±3%内"
        
        # 4. 检查小阳线 (收盘 > 开盘, 且涨幅在 0.5%~3%)
        if not is_small_positive_line(row, pre_close):
            return False, f"第{i+1}天不符合小阳线条件"
            
        pre_close = row['close'] # 更新前收盘价

    return True, "符合买入条件"

# --- 核心函数：策略执行 (与之前版本相同) ---

def run_strategy(trade_date_str):
    """
    执行策略主函数。trade_date_str 是买入日（T日，即图片中的“第4天”）。
    """
    print(f"\n--- 策略运行日期 (买入日 T): {trade_date_str} ---")
    
    # 1. 筛选并过滤股票池
    all_filtered_stocks = get_all_stocks_and_filter()
    
    buy_signals = []
    
    print("\n>>> 开始检查整理形态...")
    
    trade_date_dt = pd.to_datetime(trade_date_str, format='%Y%m%d')
    
    for ts_code in all_filtered_stocks:
        df_kline = load_stock_data(ts_code)
        
        if df_kline is None or df_kline.empty:
            continue
            
        # 筛选出日期小于或等于 T 日的所有数据
        df_recent = df_kline[df_kline.index <= trade_date_dt]
        
        # 取最近的 4 个交易日 (即 T-4, T-3, T-2, T-1)
        recent_4_days = df_recent.tail(4) 
        
        # 确保有足够的 K 线数据
        if len(recent_4_days) < 4:
            continue
            
        is_buy, reason = check_consolidation_criteria(recent_4_days)
        
        if is_buy:
            buy_signals.append({'code': ts_code, 'buy_date': trade_date_str})
            print(f"**[买入信号]** 股票代码: {ts_code}")
            
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

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_date_str = sys.argv[1]
    else:
        # 默认使用昨天的日期作为策略的买入日 T (假设今天运行，看昨天的收盘数据)
        run_date_str = (date.today() - timedelta(days=1)).strftime('%Y%m%d')

    run_strategy(run_date_str)
