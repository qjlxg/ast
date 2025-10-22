import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Define the directory where historical stock data is located
STOCK_DATA_DIR = 'stock_data'

# --- 信号中文映射表 ---
SIGNAL_CN_MAP = {
    'MACD_Golden_Cross': 'MACD金叉',
    'MACD_Bar_Turned_Positive': 'MACD柱转正',
    'KDJ_Oversold_Golden_Cross': 'KDJ超卖金叉',
    'Price_Cross_MA5': '价格上穿MA5',
    'MA_5_20_Golden_Cross': 'MA5-MA20金叉',
    'RSI_Rising_Strongly': 'RSI强劲上涨'
}
# --------------------

# --- Function to fetch historical data from local files ---
def fetch_stock_ohlc(stock_code, stock_name, end_date):
    """
    Loads OHLCV historical data from the local file system based on the stock code.
    
    :param stock_code: Stock code (e.g., '000001')
    :param stock_name: Stock name (e.g., '平安银行')
    :param end_date: Screening cut-off date
    :return: DataFrame with OHLCV data or an empty DataFrame if file is not found/invalid.
    """
    file_path = os.path.join(STOCK_DATA_DIR, f'{stock_code}.csv')
    
    try:
        # Attempt to read the file, assuming UTF-8 encoding
        df = pd.read_csv(file_path, encoding='utf-8', dtype={'股票代码': str})
        
        # Match the uploaded CSV file format and rename columns for calculation
        df = df.rename(columns={
            '日期': 'Date', 
            '开盘': 'Open', 
            '收盘': 'Close', 
            '最高': 'High', 
            '最低': 'Low', 
            '成交量': 'Volume'
        })
        
        # Convert Date format and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        
        # Filter data up to the end date and take the last 60 trading days for sufficient lookback
        df = df[df.index <= pd.to_datetime(end_date)].tail(60) 

        # Ensure required columns exist and convert them to numeric types
        required_cols = ['Open', 'Close', 'High', 'Low']
        for col in required_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                # 中文提示
                print(f"警告: 股票 {stock_code} 的 '{col}' 列无法转换为数字。跳过。")
                return pd.DataFrame()
                
            if col not in df.columns or df[col].isnull().all():
                # 中文提示
                print(f"警告: 股票 {stock_code} 文件缺少必需的列: {col} 或数据无效。跳过。")
                return pd.DataFrame()
        
        # Add stock information
        df['StockCode'] = stock_code
        df['StockName'] = stock_name
        
        # Filter out weekend data (dayofweek > 4 means Saturday or Sunday)
        return df.drop(df[df.index.dayofweek > 4].index)

    except FileNotFoundError:
        # 中文提示
        print(f"错误: 找不到股票 {stock_code} 的历史数据文件 {file_path}。跳过。")
        return pd.DataFrame()
    except Exception as e:
        # 中文提示
        print(f"错误: 读取和处理股票 {stock_code} 数据时发生异常: {e}")
        return pd.DataFrame()


# --- Technical Indicator Calculation Function ---
def calculate_all_indicators(df):
    """Calculates MA, MACD, KDJ, and RSI indicators."""
    
    # 1. Moving Averages (MA)
    df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['MA60'] = df['Close'].rolling(window=60, min_periods=1).mean()

    # 2. MACD Indicator
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = df['EMA12'] - df['EMA26']
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2 

    # 3. KDJ Indicator (Parameters 9, 3, 3)
    n = 9
    df['Lown'] = df['Low'].rolling(window=n).min()
    df['Highn'] = df['High'].rolling(window=n).max()
    # Calculate RSV (Raw Stochastic Value), handling division by zero
    rsv_divisor = (df['Highn'] - df['Lown']).replace(0, np.nan)
    df['RSV'] = ((df['Close'] - df['Lown']) / rsv_divisor * 100).fillna(0)

    m1 = 3
    df['K'] = df['RSV'].ewm(span=m1, adjust=False).mean()
    m2 = 3
    df['D'] = df['K'].ewm(span=m2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    # 4. RSI Indicator (Parameter 14)
    period = 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=period, adjust=False).mean()
    # Handle division by zero for RS
    rs_divisor = loss.replace(0, np.nan)
    RS = gain / rs_divisor
    df['RSI'] = 100 - (100 / (1 + RS))
    
    return df

# --- Screening Logic Function ---
def get_uptrend_signals(df):
    """Returns the signal score and details based on common bullish technical characteristics."""
    
    signals = {}
    
    # Ensure there is enough data for analysis (e.g., at least 20 data points)
    if len(df) < 20: 
        return 0, {}

    latest = df.iloc[-1]
    second_latest = df.iloc[-2]

    # A. Momentum Signals (MACD & KDJ)
    # MACD Golden Cross: DIF crosses above DEA today
    if latest['DIF'] > latest['DEA'] and second_latest['DIF'] <= second_latest['DEA']:
        signals['MACD_Golden_Cross'] = True
        
    # MACD Bar turns positive: MACD bar (DIF-DEA)*2 crosses above 0
    if latest['MACD'] > 0 and second_latest['MACD'] <= 0:
        signals['MACD_Bar_Turned_Positive'] = True
        
    # KDJ Oversold Golden Cross: K crosses above D, and J was recently in oversold area (<30)
    if latest['K'] > latest['D'] and second_latest['K'] <= second_latest['D'] and second_latest['J'] < 30:
        signals['KDJ_Oversold_Golden_Cross'] = True
        
    # B. Trend Signals (MA & Price)
    # Price crosses above MA5: Close price crosses above MA5 today
    if latest['Close'] > latest['MA5'] and second_latest['Close'] <= second_latest['MA5']:
        signals['Price_Cross_MA5'] = True
    
    # MA5 crosses above MA20: Short-term MA crosses mid-term MA today
    if latest['MA5'] > latest['MA20'] and second_latest['MA5'] <= second_latest['MA20']:
        signals['MA_5_20_Golden_Cross'] = True

    # C. Strength Signal (RSI)
    # RSI Rising Strongly: RSI is above 50 and has been rising over the last 5 days
    if latest['RSI'] > 50 and latest['RSI'] > df.iloc[-5]['RSI']:
        signals['RSI_Rising_Strongly'] = True

    score = len(signals)
    return score, signals

# --- Main Execution Function ---
def main(date_str=None):
    # 1. Set Date and File Paths
    if date_str is None:
        today = datetime.now()
        date_str = today.strftime('%Y%m%d')
    else:
        # Use provided date string (e.g., '20251022')
        try:
            today = datetime.strptime(date_str, '%Y%m%d')
        except ValueError:
            # 中文提示
            print(f"错误: 提供的日期格式无效: {date_str}。预期格式为 YYYYMMDD。")
            sys.exit(1)

    input_file_path = f'buy_signals/{date_str}/{date_str}.csv'
    output_dir = f'screened_signals/{date_str}'
    output_file_path = f'{output_dir}/{date_str}_screened.csv'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 中文控制台输出
    print(f"--- 启动股票筛选器，日期: {date_str} ---")
    print(f"输入文件路径: {input_file_path}")
    print(f"历史数据目录: {STOCK_DATA_DIR}")

    # 2. Read Input CSV File
    try:
        # Load the list of stocks to screen
        input_df = pd.read_csv(input_file_path, dtype={'StockCode': str})
    except FileNotFoundError:
        # 中文错误提示
        print(f"错误: 找不到输入文件 {input_file_path}。该文件必须存在且每日更新。")
        sys.exit(1)
    except Exception as e:
        # 中文错误提示
        print(f"错误: 读取输入文件 {input_file_path} 时发生未知错误: {e}")
        sys.exit(1)
        
    results = []
    
    # 3. Iterate through stocks, fetch data, calculate indicators, and screen
    for index, row in input_df.iterrows():
        code = row.get('StockCode', row.get('股票代码'))
        name = row.get('StockName', row.get('股票名称', code))
        
        if not code:
            # 中文警告提示
            print(f"警告: 缺少股票代码，跳过行 {index}。")
            continue

        # Load historical data from local file
        stock_data = fetch_stock_ohlc(code, name, today)
        
        if stock_data.empty:
            continue
        
        # Calculate indicators
        analyzed_data = calculate_all_indicators(stock_data)
        
        # Screen for signals
        score, signals = get_uptrend_signals(analyzed_data)
        
        # 将英文信号名转换为中文并连接
        chinese_signals = [SIGNAL_CN_MAP.get(sig, sig) for sig in signals.keys()] 
        
        # Record results (使用中文作为 DataFrame 列名)
        results.append({
            '股票代码': code,
            '股票名称': name,
            '评分': score,
            '信号数量': len(signals),
            '信号详情': ', '.join(chinese_signals), # 信号内容也为中文
            '最新收盘价': analyzed_data['Close'].iloc[-1] if not analyzed_data.empty else None
        })

        # 中文控制台输出
        print(f"已处理 {name} ({code})，评分: {score}")

    # 4. Finalize and Output Results
    output_df = pd.DataFrame(results)
    
    # Filter for stocks with signals (Score > 0)
    screened_df = output_df[output_df['评分'] > 0].copy() # 使用中文列名
    
    # Sort by Score (descending) and LatestClose (descending)
    screened_df = screened_df.sort_values(by=['评分', '最新收盘价'], ascending=[False, False]) # 使用中文列名
    
    # Save results to CSV file
    if not screened_df.empty:
        # 确保 CSV 以 UTF-8 编码保存，以正确显示中文
        screened_df.to_csv(output_file_path, index=False, encoding='utf8')
        # 中文控制台输出
        print(f"\n--- 筛选完成 ---")
        print(f"已筛选的股票保存到: {output_file_path}")
        print(screened_df)
    else:
        # 中文控制台输出
        print("\n--- 筛选完成 ---")
        print("没有股票符合看涨的技术特征标准。")

if __name__ == '__main__':
    # Determine the date string based on command line argument or default
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        # Default date for testing (e.g., if manually run without arguments)
        main('20251022')
