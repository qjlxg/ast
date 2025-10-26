import akshare as ak  # 仅用于获取股票名称
import pandas as pd
import numpy as np
from datetime import datetime
import os
import asyncio
import mplfinance as mpf
import nest_asyncio

# 启用 nest_asyncio
nest_asyncio.apply()

# 名称缓存文件路径
STOCK_NAMES_CACHE_FILE = 'stock_names_cache.csv'

# 设置 MPLFINANCE 的中文字体（在 Linux 环境下，例如 GitHub Actions）
# 推荐使用 simhei, 如果环境没有，则使用 WenQuanYi Zen Hei
MPF_FONT_NAME = 'SimHei' 
MPF_FONT_SIZE = 12

# --- 全局结果路径配置 ---

# 获取当前日期和时间 (已在 YAML 中设置 TZ=Asia/Shanghai)
CURRENT_DATETIME = datetime.now()
# 年月目录 (例如: 202510)
DATE_MONTH_DIR = CURRENT_DATETIME.strftime('%Y%m')
# 文件名前缀 (例如: 20251020_204702)
DATE_TIME_PREFIX = CURRENT_DATETIME.strftime('%Y%m%d_%H%M%S')
# 结果保存的根目录
RESULTS_ROOT_DIR = 'screening_results'
# 当次运行结果保存的年月目录
OUTPUT_DIR = os.path.join(RESULTS_ROOT_DIR, DATE_MONTH_DIR)
# K线图保存的子目录
K_LINE_PLOTS_DIR = os.path.join(OUTPUT_DIR, 'k_line_plots')
# 错误日志文件名
ERROR_LOG_FILE = os.path.join(OUTPUT_DIR, f'{DATE_TIME_PREFIX}_error_log.txt')
# 筛选结果文件名
SELECTED_STOCKS_FILE = os.path.join(OUTPUT_DIR, f'{DATE_TIME_PREFIX}_selected_stocks.csv')
# 筛选结果文件名 (中间/临时文件，文件名和最终结果相同)
SELECTED_STOCKS_INTERMEDIATE_FILE = os.path.join(OUTPUT_DIR, f'{DATE_TIME_PREFIX}_selected_stocks_intermediate.csv')


# --- 辅助函数 ---

def get_stock_name_map(local_codes):
    """
    1. 尝试从本地缓存文件读取股票代码-名称映射。
    2. 如果缓存文件不存在或包含未匹配的代码，则使用 akshare 从网络更新 (使用更稳定的接口)。
        - 尝试使用全面的 A 股列表。
        - 尝试使用次新股和新股列表补充。
        - 增加备用接口 (stock_zh_a_spot_em) 提高鲁棒性。
    3. 返回最终的名称映射表，并更新缓存文件。
    """
    name_map = {}
    codes_to_fetch = set(local_codes)
    all_ak_map = {} # 用于存储所有 akshare 获取的最新数据，以便写入缓存

    # 1. 尝试从缓存读取
    if os.path.exists(STOCK_NAMES_CACHE_FILE):
        try:
            # 确保使用正确的编码读取
            cache_df = pd.read_csv(STOCK_NAMES_CACHE_FILE, dtype={'code': str}, encoding='utf-8-sig')
            if 'code' in cache_df.columns and 'name' in cache_df.columns:
                
                cache_df['code'] = cache_df['code'].str.zfill(6)
                cached_map = cache_df.set_index('code')['name'].to_dict()
                all_ak_map.update(cached_map)
                
                for code in local_codes:
                    if code in cached_map and cached_map[code] != '未知': 
                        name_map[code] = cached_map[code]
                        codes_to_fetch.discard(code) 
                        
                print(f"[{datetime.now()}] [INFO] Loaded {len(cached_map)} names from cache. {len(codes_to_fetch)} codes need update.", flush=True)

        except Exception as e:
            print(f"[{datetime.now()}] [WARNING] Failed to read cache {STOCK_NAMES_CACHE_FILE}: {str(e)}. Will refresh all.", flush=True)
            codes_to_fetch = set(local_codes) 
            name_map = {}

    # 2. 如果有代码缺失或缓存失败，则从 akshare 获取最新全市场数据
    if codes_to_fetch or not os.path.exists(STOCK_NAMES_CACHE_FILE):
        
        # 2.1 尝试使用全面的 A 股列表 (stock_info_a_code_name)
        try:
            print(f"[{datetime.now()}] [INFO] Fetching stock names from akshare (via stock_info_a_code_name)...", flush=True)
            all_a_stocks = ak.stock_info_a_code_name() 
            all_a_stocks['code'] = all_a_stocks['code'].astype(str).str.zfill(6)
            ak_map_main = all_a_stocks.set_index('code')['name'].to_dict()
            all_ak_map.update(ak_map_main)
            print(f"[{datetime.now()}] [INFO] Akshare fetched {len(ak_map_main)} names from main list.", flush=True)

        except Exception as e:
            print(f"[{datetime.now()}] [ERROR] Failed to fetch main stock names from akshare: {str(e)}", flush=True)
        
        # 2.2 尝试使用次新股和新股列表补充 (stock_zh_a_new_code_name)
        try:
            print(f"[{datetime.now()}] [INFO] Fetching names from akshare (via stock_zh_a_new_code_name)...", flush=True)
            new_stocks = ak.stock_zh_a_new_code_name()
            # 字段名可能不同，这里使用 '代码' 和 '名称'
            new_stocks['code'] = new_stocks['代码'].astype(str).str.zfill(6)
            ak_map_new = new_stocks.set_index('code')['名称'].to_dict()
            all_ak_map.update(ak_map_new) 
            print(f"[{datetime.now()}] [INFO] Akshare fetched {len(ak_map_new)} names from new list.", flush=True)
            
        except Exception as e:
            print(f"[{datetime.now()}] [ERROR] Failed to fetch new stock names from akshare: {str(e)}", flush=True)

        # 2.3 增加备用接口：使用 stock_zh_a_spot_em 获取实时股票数据补充名称
        try:
            print(f"[{datetime.now()}] [INFO] Fetching names from akshare (via stock_zh_a_spot_em)...", flush=True)
            spot_stocks = ak.stock_zh_a_spot_em()
            spot_stocks['code'] = spot_stocks['代码'].astype(str).str.zfill(6)
            ak_map_spot = spot_stocks.set_index('code')['名称'].to_dict()
            all_ak_map.update(ak_map_spot)
            print(f"[{datetime.now()}] [INFO] Akshare fetched {len(ak_map_spot)} names from spot list.", flush=True)
        except Exception as e:
            print(f"[{datetime.now()}] [ERROR] Failed to fetch spot stock names from akshare: {str(e)}", flush=True)

        # 3. 合并新获取的名称到 name_map 中 (只更新需要更新的代码)
        fetched_codes_count = 0
        for code in codes_to_fetch.copy(): 
            fetched_name = all_ak_map.get(code)
            if fetched_name:
                name_map[code] = fetched_name
                codes_to_fetch.discard(code)
                fetched_codes_count += 1
                
        print(f"[{datetime.now()}] [INFO] Successfully matched {fetched_codes_count} names from Akshare network fetch.", flush=True)

        # 4. 更新缓存文件 (写入所有 akshare 获取到的最新数据)
        if all_ak_map:
            new_cache_df = pd.DataFrame(list(all_ak_map.items()), columns=['code', 'name'])
            new_cache_df['code'] = new_cache_df['code'].astype(str).str.zfill(6)
            new_cache_df.drop_duplicates(subset=['code'], keep='last', inplace=True)
            new_cache_df.to_csv(STOCK_NAMES_CACHE_FILE, index=False, encoding='utf-8-sig')
            print(f"[{datetime.now()}] [INFO] Cache file {STOCK_NAMES_CACHE_FILE} updated with {len(new_cache_df)} entries.", flush=True)
        else:
            print(f"[{datetime.now()}] [WARNING] Akshare fetch failed entirely. Cache not updated.", flush=True)

    # 5. 确保所有本地代码都有名称
    final_map = {}
    for code in local_codes:
        # 优先使用 name_map (缓存+新获取的)，否则尝试使用 all_ak_map (仅新获取的)，最后是 '未知'
        final_map[code] = name_map.get(code) or all_ak_map.get(code) or '未知'
        
    # 6. 记录仍未匹配的代码
    unmatched_codes = [code for code in local_codes if final_map[code] == '未知']
    if unmatched_codes:
        print(f"[{datetime.now()}] [WARNING] {len(unmatched_codes)} codes still unmatched: {unmatched_codes}", flush=True)
        with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now()}] [WARNING] Unmatched codes: {unmatched_codes}\n")
    
    return final_map


def get_stock_list():
    """
    1. 扫描 'stock_data' 目录下的所有 CSV 文件，获取本地股票代码列表。
    2. 使用缓存/akshare 获取的名称映射表匹配本地代码。
    """
    stock_data_dir = 'stock_data'
    if not os.path.exists(stock_data_dir):
        print(f"[{datetime.now()}] [ERROR] Directory '{stock_data_dir}' not found. Cannot load stock list.", flush=True)
        return pd.DataFrame(columns=['ts_code', 'name'])

    
    # 1. 扫描目录下的所有 .csv 文件，获取本地代码列表
    files = [f for f in os.listdir(stock_data_dir) if f.endswith('.csv')]
    local_codes = set(f.replace('.csv', '') for f in files)
    
    if not local_codes:
        print(f"[{datetime.now()}] [WARNING] No valid CSV files found in 'stock_data' directory.", flush=True)
        return pd.DataFrame(columns=['ts_code', 'name'])

    # 2. 获取名称映射表
    name_map = get_stock_name_map(local_codes)
    
    # 3. 匹配代码和名称
    stock_list_data = []
    for code in local_codes:
        stock_name = name_map.get(code, '未知')
        
        # 根据股票代码前缀确定 ts_code 格式
        ts_code = f"{code}.SS" if code.startswith('6') else f"{code}.SZ"
        
        stock_list_data.append({'ts_code': ts_code, 'name': stock_name})

    stock_list = pd.DataFrame(stock_list_data).drop_duplicates(subset=['ts_code'])
    
    print(f"[{datetime.now()}] [INFO] Local directory successfully matched {len(stock_list)} stocks with names.", flush=True)
    
    return stock_list[['ts_code', 'name']]


def is_limit_up(close, pre_close):
    """判断是否涨停 (涨幅 >=9.9%)"""
    if pre_close <= 0:
        return False
    return (close / pre_close - 1) >= 0.099


# --- 重点修改区域：新的绘图函数，实现多指标子图 ---
def plot_analysis_chart(df, ts_code, name, signal_data, plot_dir):
    """
    绘制带有 K 线、均线、信号点、MACD 和 RSI 的图表，并保存到文件。
    
    参数:
        df (pd.DataFrame): 股票数据，包含 '开盘', '收盘', '最高', '最低', '成交量', 'MACD', 'RSI' 等。
        ts_code (str): 股票代码 (带后缀)。
        name (str): 股票名称。
        signal_data (dict): 包含 '突破前高价' 和 '突破日' 的信号信息。
        plot_dir (str): K线图保存的目录。
    """
    try:
        # 1. 准备数据
        
        # 确保列名符合 mplfinance 要求
        df.rename(columns={
            '开盘': 'Open', '收盘': 'Close', '最高': 'High', '最低': 'Low', '成交量': 'Volume'
        }, inplace=True)
        
        # 将日期设置为索引
        if not isinstance(df.index, pd.DatetimeIndex):
             df.index = pd.to_datetime(df.index)
        
        # 截取用于绘图的数据，例如近 60 个交易日
        plot_df = df.iloc[-60:].copy()

        # 2. 准备绘图配置 (均线mav=(5, 10)直接在mpf.plot中设置)
        
        # MACD: 计算 Hist (DIFF - DEA)
        plot_df['MACD_Hist'] = plot_df['MACD'] - plot_df['Signal']
        
        # Addplots 配置: MACD 和 RSI
        apds = [
            # MACD 柱状图 (Panel 2)
            mpf.make_addplot(plot_df['MACD_Hist'], type='bar', color=np.where(plot_df['MACD_Hist'] > 0, '#cc121a', '#17a224').tolist(), 
                             width=0.7, secondary_y=False, panel=2, ylabel='MACD'),
            # MACD 快线 (DIF)
            mpf.make_addplot(plot_df['MACD'], color='#ff9900', secondary_y=False, panel=2),
            # MACD 慢线 (DEA)
            mpf.make_addplot(plot_df['Signal'], color='#0a9435', secondary_y=False, panel=2),
            
            # RSI 指标 (Panel 3)
            mpf.make_addplot(plot_df['RSI'], panel=3, ylabel='RSI(14)', color='#1f77b4'),
            # RSI 辅助线
            mpf.make_addplot([70] * len(plot_df), panel=3, color='gray', linestyle='--', alpha=0.6),
            mpf.make_addplot([30] * len(plot_df), panel=3, color='gray', linestyle='--', alpha=0.6),
        ]

        # 3. 准备信号点
        
        # 信号点位于突破日当天
        signal_date_str = str(signal_data['突破日'])
        signal_date = pd.to_datetime(signal_date_str, format='%Y%m%d')
        
        scatters = []
        if signal_date in plot_df.index:
            # 突破日当天的 K 线数据 (收盘价)
            close_price = plot_df.loc[signal_date]['Close']
            
            # 创建 scatter plot marker (在收盘价上方标记)
            scatters = [
                mpf.make_addplot(
                    plot_df.index[plot_df.index == signal_date], 
                    type='scatter', 
                    markersize=150, 
                    marker='^', 
                    color='red', 
                    y=close_price * 1.01 # 标记在收盘价上方
                )
            ]
            apds.extend(scatters)

        # 4. 设置风格
        
        # 保持涨跌颜色
        mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
        # 确保使用中文字体
        s = mpf.make_mpf_style(
            marketcolors=mc, 
            figcolor='white', 
            gridcolor='#e0e0e0', 
            y_on_right=False,
            rc={'font.sans-serif': MPF_FONT_NAME, 
                'font.size': MPF_FONT_SIZE, 
                'axes.unicode_minus': False}
        )
        
        # 5. 绘图
        
        # 文件名格式：代码_名称_analysis.png
        code_no_suffix = ts_code.split('.')[0]
        filename = os.path.join(plot_dir, f"{code_no_suffix}_{name}_analysis.png")
        title = f"{ts_code} {name} (突破前高: {signal_data['突破前高价']} @ {signal_data['突破日']})"
        
        # panel_ratios: K线(5), 成交量(2), MACD(2), RSI(2)
        mpf.plot(
            plot_df, 
            type='candle', 
            mav=(5, 10), # 绘制 5 日和 10 日均线
            volume=True, # 绘制成交量 (Panel 1)
            show_nontrading=False,
            addplot=apds,
            title=title,
            style=s, 
            savefig=dict(fname=filename, dpi=100),
            figratio=(16, 9),  
            panel_ratios=(5, 2, 2, 2) # 主图, 成交量, MACD, RSI 的高度比例
        )
        print(f"[{datetime.now()}] K-Line plot saved: {filename}", flush=True)

    except Exception as e:
        error_msg = f"Error plotting K-line for {ts_code} {name}: {e}"
        print(f"[{datetime.now()}] {error_msg}", flush=True)
        # 记录错误到日志文件
        with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now()}] {error_msg}\n")
        return

# 移除旧的 plot_k_line 函数，用新的 plot_analysis_chart 替换

def fetch_stock_data_sync(code, *args, **kwargs):
    """
    同步获取股票日线数据，从本地 stock_data 目录读取 CSV 文件。
    """
    file_path = os.path.join('stock_data', f'{code}.csv')
    
    if not os.path.exists(file_path):
        return code, None, f"Local data file not found: {file_path}"
    
    try:
        # 使用更灵活的编码读取，如果 utf-8 失败，尝试 gbk
        try:
            df = pd.read_csv(file_path, parse_dates=['日期'], encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, parse_dates=['日期'], encoding='gbk')
        
        # 统一列名检查
        required_cols = ['日期', '开盘', '最高', '最低', '收盘', '成交量']
        
        if '股票代码' in df.columns:
            df = df.rename(columns={'股票代码': 'code'})
        
        if not all(col in df.columns for col in required_cols):
            return code, None, f"Local data file missing required columns. Found: {list(df.columns)}"
        
        if not df.empty and '日期' in df.columns and '收盘' in df.columns:
            df = df[required_cols]
            return code, df, None
        
        return code, None, "Local data file is empty or incomplete."

    except Exception as e:
        return code, None, f"Error reading local data file {file_path}: {str(e)}"


def fetch_data_and_pre_check(code, *args, **kwargs):
    """
    同步获取股票日线数据 (从本地)，并进行 1-20 日涨停预检查。
    """
    
    code, df, error = fetch_stock_data_sync(code)
    
    if error or df.empty or len(df) < 21:
        return code, False, error
    
    try:
        df = df.sort_values('日期').reset_index(drop=True)
        
        for i in range(1, 21): 
            
            if len(df) <= i:
                continue
                
            curr_day = df.iloc[-i] 
            if len(df) <= i + 1:
                continue

            prev_day = df.iloc[-i - 1] 
            
            if is_limit_up(curr_day['收盘'], prev_day['收盘']):
                return code, True, None # 预筛选通过
        
        return code, False, None # 未通过预筛选
        
    except Exception as e:
        return code, False, f"Pre-check logic error: {str(e)}"

# --- 核心筛选函数 (两阶段 - 已修改 BB 条件) ---

# RSI 计算辅助函数
def calculate_rsi(data, window=14):
    diff = data.diff(1).dropna()
    gain = diff.apply(lambda x: x if x > 0 else 0)
    loss = diff.apply(lambda x: -x if x < 0 else 0)
    avg_gain = gain.ewm(span=window, adjust=False).mean()
    avg_loss = loss.ewm(span=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
    
async def screen_stocks(stock_list, days=30, min_days=20):
    
    # =========================================================
    # PHASE 1: Pre-Screening
    # =========================================================
    
    if stock_list.empty:
        print(f"[{datetime.now()}] Stock list is empty. Screening skipped.", flush=True)
        return pd.DataFrame()

    print(f"\n[{datetime.now()}] --- PHASE 1: Pre-screening {len(stock_list)} stocks for 1-20 day limit-up ---", flush=True)
    
    pre_screen_tasks = []
    
    for _, stock in stock_list.iterrows():
        code = stock['ts_code'].split('.')[0]
        task = asyncio.to_thread(fetch_data_and_pre_check, code) 
        pre_screen_tasks.append(task)
        
    candidate_codes = []
    error_log_p1 = []
    
    for index, future in asyncio.as_completed(pre_screen_tasks):
        code, passed, error = await future
        
        if passed:
            candidate_codes.append(code)
        
        if (index + 1) % 100 == 0 or index == len(stock_list) - 1:
            print(f"[{datetime.now()}] [PROGRESS-P1] Checked {index + 1}/{len(stock_list)}. Candidates found: {len(candidate_codes)}", flush=True)
            
        if error:
            error_log_p1.append(f"Error in pre-check for {code}: {error}")

    candidate_stock_list = stock_list[stock_list['ts_code'].str.split('.').str[0].isin(candidate_codes)].copy()
    
    print(f"[{datetime.now()}] --- PHASE 1 COMPLETED. Found {len(candidate_stock_list)} candidates. ---", flush=True)

    if candidate_stock_list.empty:
        print(f"[{datetime.now()}] No stocks passed the 1-20 day limit-up pre-screen. Screening finished.", flush=True)
        with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
            for log in error_log_p1:
                f.write(f"[{datetime.now()}] [P1 Error] {log}\n")
        return pd.DataFrame()

    # =========================================================
    # PHASE 2: Detailed Screening
    # =========================================================
    
    print(f"\n[{datetime.now()}] --- PHASE 2: Detailed screening on {len(candidate_stock_list)} candidates ---", flush=True)
    
    detailed_screen_tasks = []
    selected = []
    error_log_p2 = []
    
    for _, stock in candidate_stock_list.iterrows():
        code = stock['ts_code'].split('.')[0]
        task = asyncio.to_thread(fetch_stock_data_sync, code) 
        detailed_screen_tasks.append(task)
        
    total_candidates = len(candidate_stock_list)
    
    for index, future in asyncio.as_completed(detailed_screen_tasks):
        code, df, error = await future
        
        ts_code = f"{code}.SS" if code.startswith('6') else f"{code}.SZ"
        name = candidate_stock_list.set_index('ts_code').loc[ts_code, 'name'] if ts_code in candidate_stock_list.set_index('ts_code').index else '未知'

        status_msg = f"[{datetime.now()}] [PROGRESS-P2] {index+1}/{total_candidates} | Checking {ts_code} ({name})"
        
        if error:
            error_log_p2.append(f"Error processing {ts_code}: {error}")
            print(f"{status_msg} - FAILED (Data Error)", flush=True)
            continue
            
        if df.empty or len(df) < days + 26: # 需要足够的历史数据计算 MACD 和 RSI
            error_log_p2.append(f"Skipping {ts_code}: Insufficient data ({len(df)} days). Required: {days + 26}")
            print(f"{status_msg} - SKIPPED (Data Insufficient)", flush=True)
            continue
        
        try:
            # --- 数据准备及指标计算 ---
            df = df.sort_values('日期').reset_index(drop=True)
            df['日期'] = pd.to_datetime(df['日期'])
            df['trade_date'] = df['日期'].dt.strftime('%Y%m%d')
            
            # 计算指标 (MACD, RSI 必须在完整的 DF 上计算)
            # MACD (默认参数: 快线12, 慢线26, 信号线9)
            EMA_12 = df['收盘'].ewm(span=12, adjust=False).mean()
            EMA_26 = df['收盘'].ewm(span=26, adjust=False).mean()
            df['MACD'] = EMA_12 - EMA_26
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # RSI (默认参数: 14)
            df['RSI'] = calculate_rsi(df['收盘'], window=14)
            
            # 基础指标
            df['VOL_MA5'] = df['成交量'].rolling(5).mean()
            df['MA5'] = df['收盘'].rolling(5).mean()
            df['MA10'] = df['收盘'].rolling(10).mean()
            df['pre_high_20'] = df['最高'].rolling(window=20, closed='left').max()
            # 20 日成交量均线
            df['VOL_MA20'] = df['成交量'].rolling(20).mean()
            # 布林带 (20 日均线 ± 2 标准差)
            df['BB_MIDDLE'] = df['收盘'].rolling(20).mean()
            df['BB_STD'] = df['收盘'].rolling(20).std()
            df['BB_UPPER'] = df['BB_MIDDLE'] + 2 * df['BB_STD']
            df['BB_LOWER'] = df['BB_MIDDLE'] - 2 * df['BB_STD']
            
            # 截取近 days+1 天数据进行分析
            df_slice = df.iloc[-(days + 1):].copy().reset_index(drop=True)
            df_recent = df_slice.iloc[1:].copy() 
            df_recent['Pre_Close'] = df_slice['收盘'].shift(1).iloc[1:] 
            df_recent['pre_high_20'] = df_slice['pre_high_20'].iloc[1:]
            
            curr = df_recent.iloc[-1]
            signal = None
            limit_up_dates = []
            pre_high_val = None
            limit_up_date_val = None
            
            # --- 统一战法：找 D-1 到 D-20 内最近的“涨停突破前高”日，并检查当前回踩信号 ---
            df_recent['Is_Limit_Up'] = df_recent.apply(lambda row: is_limit_up(row['收盘'], row['Pre_Close']), axis=1)
            trigger_day_data = None
            
            for i in range(2, min(22, len(df_recent) + 1)): 
                lu_day = df_recent.iloc[-i]
                
                if lu_day['Is_Limit_Up']:
                    # 突破日条件增强：涨停 + 突破前高 + 成交量 > 20日均量 (放量突破)
                    if (lu_day['收盘'] > lu_day['pre_high_20']) and \
                       (lu_day['成交量'] > lu_day['VOL_MA20']): 
                        trigger_day_data = lu_day
                        break # 找到最近的突破日即停止
            
            if trigger_day_data is not None:
                
                # ***** 修改点 1: 弱化强势确认条件 (中轨 -> 下轨) *****
                # 筛选条件 1：当前收盘价必须在布林带下轨之上确认未崩盘
                if curr['收盘'] >= curr['BB_LOWER']:
                    
                    # 筛选条件 2：检查突破日到昨天，回踩过程中收盘价是否跌破布林带下轨
                    # 保持这个严格条件不变
                    start_index = df_recent.index.get_loc(trigger_day_data.name)
                    # 检查从突破日到昨天的回踩过程中是否有弱势信号
                    if (df_recent.iloc[start_index:-1]['收盘'] < df_recent.iloc[start_index:-1]['BB_LOWER']).any():
                        print(f"{status_msg} - NO MATCH (Failed: Broke BB_LOWER during callback)", flush=True)
                        continue # 回踩过程中跌破下轨，信号取消
                    
                    # 筛选条件 3：原始缩量回踩 5 日线逻辑 (保持不变)
                    callback_shrink = False
                    for i in range(2, min(5, len(df_recent) + 1)): 
                        prev = df_recent.iloc[-i] 
                        
                        if (prev['成交量'] < prev['VOL_MA5'] * 0.8 and 
                            abs(prev['收盘'] - prev['MA5']) / prev['MA5'] <= 0.02):
                            callback_shrink = True
                            break
                            
                    if callback_shrink and curr['收盘'] >= curr['MA5'] * 0.98:
                        
                        # 修改信号描述，反映 BB 确认条件的弱化
                        signal = '买入（涨停放量突破前高 + 缩量回踩5日线 + BB下轨确认）'
                        pre_high_val = trigger_day_data['pre_high_20']
                        limit_up_date_val = trigger_day_data['trade_date']
                        limit_up_dates.append(trigger_day_data['日期'])
            
            
            if signal:
                # 构造绘图数据，包含 MACD, RSI
                df_plot = df_recent[[
                    '日期', '开盘', '最高', '最低', '收盘', '成交量', 
                    'MA5', 'MA10', 'VOL_MA20', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER',
                    'MACD', 'Signal', 'RSI' # 新增的指标
                ]].copy()
                df_plot.set_index('日期', inplace=True)
                
                # 使用新的绘图函数
                plot_analysis_chart(
                    df_plot, 
                    ts_code, 
                    name, 
                    {'突破前高价': pre_high_val, '突破日': limit_up_date_val},
                    K_LINE_PLOTS_DIR
                )

                # 使用中文列名
                selected.append({
                    '代码': ts_code, 
                    '名称': name,   
                    '当前收盘价': curr['收盘'],
                    'MA5': curr['MA5'],
                    'MA10': curr['MA10'],
                    '突破前高价': pre_high_val,
                    '突破日': limit_up_date_val,
                    '信号': signal,
                })
                
                print(f"{status_msg} - MATCHED! Signal: {signal}", flush=True) 
            else:
                print(f"{status_msg} - NO MATCH", flush=True)

        except Exception as e:
            error_log_p2.append(f"Error processing {ts_code} in screening logic: {str(e)}")
            print(f"{status_msg} - FAILED (Logic Error)", flush=True)
            continue
    
    # 统一写入所有日志文件
    all_errors = error_log_p1 + error_log_p2
    with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
        for log in all_errors:
            f.write(f"[{datetime.now()}] {log}\n") 
    
    results = pd.DataFrame(selected)
    print(f"[{datetime.now()}] --- PHASE 2 COMPLETED. {len(results)} stocks selected ---", flush=True)
    
    if not results.empty:
        # 在 CSV 中保存中文列名
        results.to_csv(SELECTED_STOCKS_FILE, index=False, encoding='utf-8-sig')
        results.to_csv(SELECTED_STOCKS_INTERMEDIATE_FILE, index=False, encoding='utf-8-sig') 
        
        print(f"\n[{datetime.now()}] 筛选结果示例 (已保存到 {SELECTED_STOCKS_FILE})：", flush=True)
        print(results.head(), flush=True)
    
    return results

if __name__ == "__main__":
    
    # 检查本地数据目录是否存在
    if not os.path.exists('stock_data'):
        print(f"[{datetime.now()}] [CRITICAL ERROR] Local 'stock_data' directory not found.", flush=True)
        print(f"[{datetime.now()}] Please create a 'stock_data' directory and place the stock CSV files inside it (e.g., stock_data/000001.csv).", flush=True)
        exit(1)

    # 确保目标目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(K_LINE_PLOTS_DIR):
        os.makedirs(K_LINE_PLOTS_DIR)

    # 启动前打印文件路径信息
    print(f"[{datetime.now()}] Results will be saved to directory: {OUTPUT_DIR}", flush=True)
    print(f"[{datetime.now()}] Final CSV file: {SELECTED_STOCKS_FILE}", flush=True)
    print(f"[{datetime.now()}] Error Log file: {ERROR_LOG_FILE}", flush=True)

    # 确保新的错误日志文件被创建
    if not os.path.exists(ERROR_LOG_FILE):
         with open(ERROR_LOG_FILE, 'w', encoding='utf-8') as f:
             f.write(f"[{datetime.now()}] --- Starting Log for Run: {DATE_TIME_PREFIX} ---\n")

    stock_list = get_stock_list()
    # 确认总股票数量
    print(f"[{datetime.now()}] Total stocks fed into Phase 1: {len(stock_list)}", flush=True)
        
    results = asyncio.run(screen_stocks(stock_list))
    if not results.empty:
        print(f"\n[{datetime.now()}] Final Result: 筛选出 {len(results)} 只符合战法的股票，保存至 {SELECTED_STOCKS_FILE}", flush=True)
    else:
        print(f"\n[{datetime.now()}] Final Result: 未找到符合条件的股票", flush=True)
