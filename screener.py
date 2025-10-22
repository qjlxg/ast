import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Define the directory where historical stock data is located
STOCK_DATA_DIR = 'stock_data'

# --- 信号中文映射表 (包含所有社区增强信号) ---
SIGNAL_CN_MAP = {
    'MACD_Golden_Cross': 'MACD金叉',
    'MACD_Turning_Positive': 'MACD柱转正',
    'KDJ_Golden_Cross_From_Oversold': 'KDJ超卖金叉',
    'Price_Above_MA5': '价格高于MA5',
    'MA_5_20_Golden_Cross': 'MA5-MA20金叉',
    'RSI_Rising_Strongly': 'RSI强劲上涨',
    'Volume_Confirm': '成交量确认',
    'BB_Low_Rebound': '布林带低位反弹',
    'OBV_Inflow': 'OBV资金流入',
    'Low_Vol_Confirm': 'ATR低波动确认'
}
# --------------------

# --- Function to fetch historical data from local files (已修复股票代码补零问题) ---
def fetch_stock_ohlc(stock_code, stock_name, end_date):
    """
    Loads OHLCV historical data from the local file system based on the stock code.
    """
    # 【修复点】确保股票代码是 6 位，前面补零
    padded_code = str(stock_code).zfill(6) 
    file_path = os.path.join(STOCK_DATA_DIR, f'{padded_code}.csv') # 使用补零后的代码
    
    try:
        # 尝试读取文件
        df = pd.read_csv(file_path, encoding='utf-8', dtype={'股票代码': str})
        
        # 重命名列
        df = df.rename(columns={
            '日期': 'Date', 
            '开盘': 'Open', 
            '收盘': 'Close', 
            '最高': 'High', 
            '最低': 'Low', 
            '成交量': 'Volume'
        })
        
        # 数据处理
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        df = df[df.index <= pd.to_datetime(end_date)].tail(60) 

        # 检查必需列
        required_cols = ['Open', 'Close', 'High', 'Low', 'Volume']
        for col in required_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                print(f"警告: 股票 {stock_code} 的 '{col}' 列无法转换为数字。跳过。")
                return pd.DataFrame()
                
            if col not in df.columns or df[col].isnull().all():
                print(f"警告: 股票 {stock_code} 文件缺少必需的列: {col} 或数据无效。跳过。")
                return pd.DataFrame()
        
        # 添加股票信息并过滤周末
        df['StockCode'] = stock_code
        df['StockName'] = stock_name
        return df.drop(df[df.index.dayofweek > 4].index)

    except FileNotFoundError:
        print(f"错误: 找不到股票 {stock_code} 的历史数据文件 {file_path}。跳过。")
        return pd.DataFrame()
    except Exception as e:
        print(f"错误: 读取和处理股票 {stock_code} 数据时发生异常: {e}")
        return pd.DataFrame()


# --- Technical Indicator Calculation Function ---
def calculate_all_indicators(df):
    """Calculates MA, MACD, KDJ, RSI, Volume MA, BB, OBV, and ATR indicators."""
    
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
    rsv_divisor = (df['Highn'] - df['Lown']).replace(0, np.nan)
    df['RSV'] = ((df['Close'] - df['Lown']) / rsv_divisor * 100).fillna(0)
    m1 = 3
    df['K'] = df['RSV'].ewm(span=m1, adjust=False).mean()
    df['D'] = df['K'].ewm(span=m1, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    # 4. RSI Indicator (14 periods)
    n_rsi = 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=n_rsi).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=n_rsi).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 5. 成交量（Volume）相关计算
    df['Vol_MA5'] = df['Volume'].rolling(window=5, min_periods=1).mean()

    # 6. 布林带（Bollinger Bands）
    df['BB_Mid'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['BB_Std'] = df['Close'].rolling(window=20, min_periods=1).std()
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']

    # 7. OBV（On-Balance Volume）
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum().fillna(0)

    # 8. ATR（Average True Range）
    df['TR'] = np.maximum(df['High'] - df['Low'], 
                           np.maximum(np.abs(df['High'] - df['Close'].shift()), 
                                      np.abs(df['Low'] - df['Close'].shift())))
    df['ATR'] = df['TR'].rolling(window=14, min_periods=1).mean()

    return df

# --- Uptrend Signal Screening Function ---
def get_uptrend_signals(df):
    """Screens for bullish technical signals based on calculated indicators."""
    
    if df.empty or len(df) < 5:
        return 0, {}
        
    signals = {}
    latest = df.iloc[-1]
    
    # --- 1. Momentum Signals (MACD, KDJ) ---
    # MACD Golden Cross: DIF crosses above DEA
    if latest['DIF'] > latest['DEA'] and df.iloc[-2]['DIF'] <= df.iloc[-2]['DEA']:
        signals['MACD_Golden_Cross'] = True
    
    # MACD Histogram Turning Positive: MACD from negative to positive
    if latest['MACD'] > 0 and df.iloc[-2]['MACD'] <= 0:
        signals['MACD_Turning_Positive'] = True
    
    # KDJ Golden Cross from Oversold: K crosses above D and J was recently below 30
    if latest['K'] > latest['D'] and df.iloc[-2]['K'] <= df.iloc[-2]['D'] and df['J'].tail(5).min() < 30:
        signals['KDJ_Golden_Cross_From_Oversold'] = True

    # --- 2. Trend Signals (MA Crosses) ---
    # Price above MA5
    if latest['Close'] > latest['MA5']:
        signals['Price_Above_MA5'] = True
    
    # MA5 crosses above MA20
    if latest['MA5'] > latest['MA20'] and df.iloc[-2]['MA5'] <= df.iloc[-2]['MA20']:
        signals['MA_5_20_Golden_Cross'] = True

    # --- 3. Strength Signals (RSI) ---
    # RSI is above 50 and has been rising over the last 5 days
    if latest['RSI'] > 50 and latest['RSI'] > df.iloc[-5]['RSI']:
        signals['RSI_Rising_Strongly'] = True

    # --- 4. Community/Real-World Signals ---

    # 成交量确认：如果现有信号触发，且最新成交量 > Vol_MA5（放量确认）
    if latest['Volume'] > latest['Vol_MA5'] and ('MACD_Golden_Cross' in signals or 'MA_5_20_Golden_Cross' in signals):
        signals['Volume_Confirm'] = True

    # 布林带低位反弹：如果Close接近BB_Lower且有RSI上升确认
    if latest['Close'] >= latest['BB_Lower'] and latest['Close'] <= latest['BB_Lower'] * 1.05 and 'RSI_Rising_Strongly' in signals:
        signals['BB_Low_Rebound'] = True

    # OBV资金流入：如果OBV高于短期均值且有RSI上升
    obv_ma5 = df['OBV'].rolling(window=5, min_periods=1).mean().iloc[-1]
    if latest['OBV'] > obv_ma5 and 'RSI_Rising_Strongly' in signals:
        signals['OBV_Inflow'] = True

    # ATR低波动确认：如果波动低且金叉
    atr_ma5 = df['ATR'].rolling(window=5, min_periods=1).mean().iloc[-1]
    if latest['ATR'] < atr_ma5 and 'MA_5_20_Golden_Cross' in signals:
        signals['Low_Vol_Confirm'] = True

    score = len(signals)
    return score, signals

# --- Main Execution Function ---
def main(date_str=None):
    # 1. Set Date and File Paths
    if date_str is None:
        # 如果未提供参数，则使用实时日期
        today = datetime.now()
        date_str = today.strftime('%Y%m%d')
    else:
        # 如果提供了参数，则使用参数指定的日期
        try:
            today = datetime.strptime(date_str, '%Y%m%d')
        except ValueError:
            print(f"错误: 提供的日期格式无效: {date_str}。预期格式为 YYYYMMDD。")
            sys.exit(1)

    input_file_path = f'buy_signals/{date_str}/{date_str}.csv'
    output_dir = f'screened_signals/{date_str}'
    output_file_path = f'{output_dir}/{date_str}_screened.csv'

    os.makedirs(output_dir, exist_ok=True)

    print(f"--- 启动股票筛选器，日期: {date_str} ---")
    print(f"输入文件路径: {input_file_path}")
    print(f"历史数据目录: {STOCK_DATA_DIR}")

    # 2. Read Input CSV File
    try:
        input_df = pd.read_csv(input_file_path, dtype={'StockCode': str})
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file_path}。该文件必须存在且每日更新。")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 读取输入文件 {input_file_path} 时发生未知错误: {e}")
        sys.exit(1)
        
    results = []
    
    # 3. Iterate through stocks, fetch data, calculate indicators, and screen
    for index, row in input_df.iterrows():
        # 获取股票代码和名称
        code = row.get('StockCode', row.get('股票代码'))
        name = row.get('StockName', row.get('股票名称', code))
        
        if not code:
            print(f"警告: 缺少股票代码，跳过行 {index}。")
            continue

        # 加载历史数据，代码会在 fetch_stock_ohlc 中自动补零
        stock_data = fetch_stock_ohlc(code, name, today)
        
        if stock_data.empty:
            continue
        
        analyzed_data = calculate_all_indicators(stock_data)
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

        print(f"已处理 {name} ({code})，评分: {score}")

    # 4. Finalize and Output Results
    output_df = pd.DataFrame(results)
    
    screened_df = output_df[output_df['评分'] > 0].copy()
    
    screened_df = screened_df.sort_values(by=['评分', '最新收盘价'], ascending=[False, False])
    
    if not screened_df.empty:
        screened_df.to_csv(output_file_path, index=False, encoding='utf8')
        print(f"\n--- 筛选完成 ---")
        print(f"已筛选的股票保存到: {output_file_path}")
        print(screened_df)
    else:
        print("\n--- 筛选完成 ---")
        print("没有股票符合看涨的技术特征标准。")

if __name__ == '__main__':
    # Determine the date string based on command line argument or default
    if len(sys.argv) > 1:
        # 如果提供了参数，则使用参数
        main(sys.argv[1])
    else:
        # 如果没有提供参数，则让 main 函数自动使用实时日期
        main()
