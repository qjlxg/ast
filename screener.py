import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Define the directory where historical stock data is located
STOCK_DATA_DIR = 'stock_data'

# --- 信号中文映射表 (更新为“四要素共振”) ---
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
    'Low_Vol_Confirm': 'ATR低波动确认',
    'Three_Elements_Resonance': '四要素共振 (筑底/均线/价格/量能)' # 更新描述
}
# --------------------

# --- Function to fetch historical data from local files (强制补零) ---
def fetch_stock_ohlc(stock_code, stock_name, end_date):
    """
    Loads OHLCV historical data from the local file system based on the stock code.
    """
    # 确保股票代码是 6 位，前面补零
    padded_code = str(stock_code).zfill(6) 
    file_path = os.path.join(STOCK_DATA_DIR, f'{padded_code}.csv')
    
    try:
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
        df['StockCode'] = padded_code
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
    """Calculates all necessary indicators, including those for filtering and scoring."""
    
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

    # 3. KDJ Indicator
    n = 9
    df['Lown'] = df['Low'].rolling(window=n).min()
    df['Highn'] = df['High'].rolling(window=n).max()
    rsv_divisor = (df['Highn'] - df['Lown']).replace(0, np.nan)
    df['RSV'] = ((df['Close'] - df['Lown']) / rsv_divisor * 100).fillna(0)
    m1 = 3
    df['K'] = df['RSV'].ewm(span=m1, adjust=False).mean()
    df['D'] = df['K'].ewm(span=m1, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    # 4. RSI Indicator
    n_rsi = 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=n_rsi).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=n_rsi).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 5. 成交量（Volume）相关计算
    df['Vol_MA5'] = df['Volume'].rolling(window=5, min_periods=1).mean()
    df['Vol_MA20'] = df['Volume'].rolling(window=20, min_periods=1).mean() 

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

    # 9. 计算昨日收盘价和前日收盘价，用于短期涨幅过滤
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev2_Close'] = df['Close'].shift(2)
    
    return df

# --- Uptrend Signal Screening Function (V6: 明确加入洗盘/筑底确认) ---
def get_uptrend_signals(df):
    """Screens for bullish technical signals based on calculated indicators and applies anti-chasing filters."""
    
    if df.empty or len(df) < 5:
        return 0, {}
        
    signals = {}
    latest = df.iloc[-1]
    
    # ----------------------------------------------------
    # 【第一步】短期超买判断 (避免追高)
    # ----------------------------------------------------
    is_overbought = False
    if len(df) >= 3:
        # 提取昨日收盘价和前日收盘价
        prev_close = df.iloc[-2]['Close']
        prev2_close = df.iloc[-3]['Close']
        
        # 计算昨日涨幅 (如果前日收盘价不为0)
        yesterday_return = (prev_close - prev2_close) / prev2_close if prev2_close != 0 else 0
        
        # 计算两天累计涨幅 (如果前日收盘价不为0)
        two_day_return = (latest['Close'] - prev2_close) / prev2_close if prev2_close != 0 else 0

        # 条件 A: 昨日已涨停 (涨幅 >= 9.5%)
        is_yesterday_limit_up = yesterday_return >= 0.095
        
        # 条件 B: 最近两天累计涨幅过高 (例如 15% 以上)
        is_two_day_soaring = two_day_return >= 0.15 
        
        # 如果满足任何一个短期超买条件，则视为超买
        if is_yesterday_limit_up or is_two_day_soaring:
            is_overbought = True

    # ----------------------------------------------------
    # 【第二步】评分计算 (仅对未超买的股票进行高分信号评估)
    # ----------------------------------------------------

    if not is_overbought:
        
        # --- 【核心信号】四要素启动 (Four Elements Launch: 筑底/均线/价格/量能) ---
        if len(df) >= 20:
            recent_data = df.tail(20)
            
            # 元素 1：筑底/洗盘确认 (Bottoming/Shakeout Confirmation)
            # 要求：在最近 20 天内，价格曾接近 MA60（长期成本线），且未有效跌破。
            ma60_avg_20d = recent_data['MA60'].mean()
            price_low_20d = recent_data['Low'].min()
            
            # 筑底条件：最低价在 MA60 均值 5% 以内 (即触及支撑)，且今天已经站上 MA20
            is_bottoming_confirmed = (price_low_20d >= ma60_avg_20d * 0.95 and  # 20日内最低价未远离 MA60
                                      latest['Close'] > latest['MA20'])      # 且今天已经站上 MA20
            
            # 元素 2：均线共振 (粘合 + 金叉)
            ma_diff_ratio_20d = (recent_data['MA5'] - recent_data['MA20']).abs() / recent_data['MA20']
            is_ma_sticky = ma_diff_ratio_20d.mean() < 0.025 # 2.5% 容忍度
            is_golden_cross = (latest['MA5'] > latest['MA20'] and 
                               df.iloc[-2]['MA5'] <= df.iloc[-2]['MA20'])
            is_ma_resonance = is_ma_sticky and is_golden_cross

            # 元素 3：价格突破 (Price Breakthrough)
            # 今天的收盘价是过去 20 个交易日（不含今天）中的最高价
            is_price_breakthrough = latest['Close'] > recent_data['Close'].iloc[:-1].max()

            # 元素 4：量能确认 (Volume Confirmation)
            # 今天的成交量显著高于最近 20 日的平均成交量（1.5 倍以上）
            is_volume_confirm = latest['Volume'] > (1.5 * latest['Vol_MA20'])
            
            # 最终四要素共振确认
            if is_bottoming_confirmed and is_ma_resonance and is_price_breakthrough and is_volume_confirm:
                signals['Three_Elements_Resonance'] = True

        # --- 其他基础信号 (用于积累评分) ---
        
        if latest['DIF'] > latest['DEA'] and df.iloc[-2]['DIF'] <= df.iloc[-2]['DEA']:
            signals['MACD_Golden_Cross'] = True
        
        if latest['MACD'] > 0 and df.iloc[-2]['MACD'] <= 0:
            signals['MACD_Turning_Positive'] = True
        
        if latest['K'] > latest['D'] and df.iloc[-2]['K'] <= df.iloc[-2]['D'] and df['J'].tail(5).min() < 30:
            signals['KDJ_Golden_Cross_From_Oversold'] = True

        if latest['Close'] > latest['MA5']:
            signals['Price_Above_MA5'] = True
        
        if latest['MA5'] > latest['MA20'] and df.iloc[-2]['MA5'] <= df.iloc[-2]['MA20']:
            signals['MA_5_20_Golden_Cross'] = True

        if latest['RSI'] > 50 and latest['RSI'] > df.iloc[-5]['RSI']:
            signals['RSI_Rising_Strongly'] = True

        if latest['Volume'] > latest['Vol_MA5'] and ('MACD_Golden_Cross' in signals or 'MA_5_20_Golden_Cross' in signals):
            signals['Volume_Confirm'] = True

        if latest['Close'] >= latest['BB_Lower'] and latest['Close'] <= latest['BB_Lower'] * 1.05 and 'RSI_Rising_Strongly' in signals:
            signals['BB_Low_Rebound'] = True

        obv_ma5 = df['OBV'].rolling(window=5, min_periods=1).mean().iloc[-1]
        if latest['OBV'] > obv_ma5 and 'RSI_Rising_Strongly' in signals:
            signals['OBV_Inflow'] = True

        atr_ma5 = df['ATR'].rolling(window=5, min_periods=1).mean().iloc[-1]
        if latest['ATR'] < atr_ma5 and 'MA_5_20_Golden_Cross' in signals:
            signals['Low_Vol_Confirm'] = True
        
    score = len(signals)
    return score, signals

# --- Main Execution Function ---
def main(date_str=None):
    # 1. Set Date and File Paths
    if date_str is None:
        today = datetime.now()
        date_str = today.strftime('%Y%m%d')
    else:
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
        input_df = pd.read_csv(input_file_path, dtype={'StockCode': str, '股票代码': str})
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file_path}。该文件必须存在且每日更新。")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 读取输入文件 {input_file_path} 时发生未知错误: {e}")
        sys.exit(1)
        
    results = []
    
    # 3. Iterate through stocks, fetch data, calculate indicators, and screen
    for index, row in input_df.iterrows():
        original_code = row.get('StockCode', row.get('股票代码'))
        name = row.get('StockName', row.get('股票名称', original_code))
        
        if not original_code:
            print(f"警告: 缺少股票代码，跳过行 {index}。")
            continue
            
        # 强制股票代码补零为 6 位
        code_padded = str(original_code).zfill(6)

        stock_data = fetch_stock_ohlc(code_padded, name, today)
        
        if stock_data.empty:
            continue
        
        analyzed_data = calculate_all_indicators(stock_data)
        score, signals = get_uptrend_signals(analyzed_data)
        
        chinese_signals = [SIGNAL_CN_MAP.get(sig, sig) for sig in signals.keys()] 
        
        results.append({
            '股票代码': code_padded, 
            '股票名称': name,
            '评分': score,
            '信号数量': len(signals),
            '信号详情': ', '.join(chinese_signals),
            '最新收盘价': analyzed_data['Close'].iloc[-1] if not analyzed_data.empty else None
        })

        print(f"已处理 {name} ({code_padded})，评分: {score}")

    # 4. Finalize and Output Results
    output_df = pd.DataFrame(results)
    
    # 只筛选评分大于 5 的股票 (满足您的要求)
    screened_df = output_df[output_df['评分'] > 5].copy()
    
    # 强制将 '股票代码' 列转换为字符串类型，以保留前导零
    screened_df['股票代码'] = screened_df['股票代码'].astype(str)
    
    # 按照评分和最新收盘价降序排列
    screened_df = screened_df.sort_values(by=['评分', '最新收盘价'], ascending=[False, False])
    
    if not screened_df.empty:
        screened_df.to_csv(output_file_path, index=False, encoding='utf8')
        print(f"\n--- 筛选完成 ---")
        print(f"已筛选的股票保存到: {output_file_path}")
        print(screened_df)
    else:
        print("\n--- 筛选完成 ---")
        print("没有股票符合看涨的技术特征标准（评分>5），或所有高分信号均因短期涨幅过大而被过滤。")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
