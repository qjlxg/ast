#=======================================================
#项目名称：精简选股信号生成器（不含沪深300指数风控）
#项目说明：专注于读取本地数据，执行个股选股逻辑，但跳过市场环境检查。
#==========================================================
import pandas as pd
import numpy as np
import talib as ta
import os
import glob
from datetime import datetime
from typing import List

# ========================================================
# 【本地数据配置】
STOCK_DATA_DIR = 'stock_data' 
INDEX_CODE = '000300.XSHG' 

# 全局变量配置
g = {
    "buy_count": 20,             # 最大推荐数量
    "universe2": [],             # 风险股票池（需手动维护）
    "GLOBAL_STOCK_LIST": [],     # 缓存所有A股代码
    # 强制设定为 normal，因为没有指数数据进行检查
    "market_status": "normal"    
}

# ========================================================
# 【日志和辅助函数】
# ... (log_info, log_warning, log_error 函数代码省略，与上个版本相同) ...
def log_info(msg):
    print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} {msg}")
def log_warning(msg):
    print(f"[WARNING] {datetime.now().strftime('%H:%M:%S')} {msg}")
def log_error(msg):
    print(f"[ERROR] {datetime.now().strftime('%H:%M:%S')} {msg}")

# ========================================================
# 【数据适配层：文件动态读取 - 保持不变】
# ** 核心文件读取函数 **
# ========================================================
def _get_stock_full_code(stock_code):
    """根据股票代码补充后缀"""
    code_prefix = stock_code.split('.')[0]
    if code_prefix.startswith('6'):
        return f"{code_prefix}.SH"
    elif code_prefix.startswith(('0', '3')):
        return f"{code_prefix}.SZ"
    return code_prefix 

def _read_and_clean_csv(file_path):
    """读取并清洗单个CSV文件"""
    try:
        df = pd.read_csv(file_path, encoding='gbk') 
        df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', 
                           '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        cols_to_keep = ['open', 'high', 'low', 'close', 'volume']
        df = df[[col for col in cols_to_keep if col in df.columns]]
        df = df.astype(float)
        return df
    except Exception:
        return pd.DataFrame()

def mock_get_data(security, end_date, frequency, fields, count):
    """模拟 get_price() 和 get_history()"""
    # 移除后缀以便查找文件
    security_no_suffix = security.split('.')[0]
    file_path = os.path.join(STOCK_DATA_DIR, f"{security_no_suffix}.csv")
    
    df = _read_and_clean_csv(file_path)
    if df.empty: return pd.DataFrame()
        
    end_dt = pd.to_datetime(end_date)
    filtered_df = df[df.index <= end_dt]
    
    if count:
        result = filtered_df.tail(count)
    else:
        result = filtered_df
        
    if fields:
        if isinstance(fields, str): fields = [fields]
        return result[fields]
        
    return result

def mock_get_Ashares():
    """获取全部A股代码列表 (扫描 stock_data 目录)"""
    if g["GLOBAL_STOCK_LIST"]:
        return g["GLOBAL_STOCK_LIST"]
        
    if not os.path.exists(STOCK_DATA_DIR):
        raise FileNotFoundError(f"数据目录 '{STOCK_DATA_DIR}' 不存在，请创建并放入CSV文件。")
        
    files = glob.glob(os.path.join(STOCK_DATA_DIR, '[0-9]*.csv'))
    stock_codes = [os.path.basename(f).replace('.csv', '') for f in files]
    
    final_list = []
    for code in stock_codes:
        if code != INDEX_CODE:
            final_list.append(_get_stock_full_code(code))

    g["GLOBAL_STOCK_LIST"] = final_list
    return final_list

def mock_get_stock_status(all_stocks, status_type, trade_date):
    """模拟获取股票状态 (简化版：仅检查当日是否有数据)"""
    status_map = {}
    dt = pd.to_datetime(trade_date)
    
    for stock in all_stocks:
        df = mock_get_data(stock, trade_date, '1d', fields=['close'], count=1)
        # 如果当日收盘价数据缺失，则视为有风险（停牌、数据缺失等）
        if df.empty or dt not in df.index:
             status_map[stock] = True 
        else:
            status_map[stock] = False
    return status_map

# ========================================================
# 【策略模块 - API替换】
# * check_market_condition 函数已移除 *
# ========================================================

def filter_risk_stocks(all_stocks, trade_date):
    """过滤ST/停牌/退市/退市整理期股票 (使用 mock_get_stock_status)"""
    st_status = mock_get_stock_status(all_stocks, 'ST', trade_date)
    stocks_to_keep = [stock for stock in all_stocks if not st_status.get(stock)]
    return stocks_to_keep 

def filter_by_volume(stock_list, trade_date, min_vol_ratio=1.0):
    """流动性过滤（成交量）(使用 mock_get_data)"""
    filtered_stocks = []
    for stock in stock_list:
        try:
            hist_data = mock_get_data(security=stock, end_date=trade_date, frequency='1d', fields=['volume'], count=30)
            if len(hist_data) < 20: continue
            volumes = hist_data['volume'].values
            recent_avg_vol = np.mean(volumes[-5:])
            long_avg_vol = np.mean(volumes)
            
            if long_avg_vol > 0 and recent_avg_vol > long_avg_vol * min_vol_ratio:
                filtered_stocks.append(stock)
        except Exception:
            pass # 忽略错误
    return filtered_stocks

def filter_long_lower_shadow(stock_list, trade_date, shadow_ratio=2.0):
    """筛选具有长下影线形态的股票 (使用 mock_get_data)"""
    filtered_stocks = []
    for stock in stock_list:
        try:
            kdata = mock_get_data(security=stock, end_date=trade_date, frequency='1d', fields=['open', 'high', 'low', 'close'], count=1)
            if kdata.empty: continue
            open_price = kdata['open'].values[0]
            high_price = kdata['high'].values[0]
            low_price = kdata['low'].values[0]
            close_price = kdata['close'].values[0]
            body_length = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            lower_shadow = body_bottom - low_price
            
            if body_length > 0 and lower_shadow > body_length * shadow_ratio:
                if (high_price - close_price) / (high_price - low_price) < 0.3:
                    filtered_stocks.append(stock)
        except Exception:
            pass # 忽略错误
    return filtered_stocks

def filter_by_rsi(stock_list, trade_date, rsi_period=14, lower_bound=30, upper_bound=70):
    """相对强度(RSI)选股 (使用 mock_get_data)"""
    filtered_stocks = []
    for stock in stock_list:
        try:
            hist_data = mock_get_data(security=stock, end_date=trade_date, frequency='1d', fields=['close'], count=30)
            if len(hist_data) < rsi_period + 5: continue
            closes = hist_data['close'].values
            rsi = ta.RSI(closes, timeperiod=rsi_period)
            current_rsi = rsi[-1]
            
            if lower_bound <= current_rsi <= upper_bound:
                filtered_stocks.append(stock)
        except Exception:
            pass # 忽略错误
    return filtered_stocks

# ========================================================
# 【核心信号生成函数】
# ========================================================
def get_daily_buy_signals(trade_date: str) -> List[str]:
    """
    运行策略的选股逻辑，输出当日符合买入条件的股票列表。
    """
    log_info("=" * 50)
    log_info(f"【信号生成开始】日期: {trade_date}")
    
    # 市场环境强制为 normal
    g["market_status"] = "normal"
    
    try:
        # 1. 获取全部A股 (扫描 stock_data 目录)
        all_stocks = mock_get_Ashares()
        log_info("原始股票池数量：%d" % len(all_stocks))
        
        # 2. 过滤风险股和手动排除的股票
        safe_stocks = [s for s in all_stocks if s not in g["universe2"]]
        safe_stocks = filter_risk_stocks(safe_stocks, trade_date)
        log_info("过滤风险股后股票数量：%d" % len(safe_stocks))   
        
        # 3. 筛选主板股票
        stock_list = [s for s in safe_stocks if s.startswith(('60', '00'))]
        log_info("筛选主板股票后数量：%d" % len(stock_list))
        
        # 4. 核心筛选链：成交量 -> 长下影线 -> RSI
        stock_list = filter_by_volume(stock_list, trade_date, min_vol_ratio=1.2)
        log_info("成交量过滤后股票数量：%d" % len(stock_list))
        
        stock_list = filter_long_lower_shadow(stock_list, trade_date, shadow_ratio=2.0)
        log_info("长下影线筛选后股票数量：%d" % len(stock_list))
        
        target_stocks = filter_by_rsi(stock_list, trade_date, rsi_period=14, lower_bound=40, upper_bound=65)
        log_info("RSI筛选后股票数量：%d" % len(target_stocks))
        
        # 5. 备选逻辑
        if len(target_stocks) < 20 and stock_list:
            # 选前20只作为备选，如果最终筛选结果少于20
            target_stocks = stock_list[:min(len(stock_list), 20)] 
            log_warning("筛选股票不足，使用长下影线池的前20只作为备选。")

        log_info("【信号生成完成】今日推荐买入股票数：%d" % len(target_stocks))
        log_info("推荐买入股票清单：%s" % target_stocks)
        
        return target_stocks
          
    except Exception as e:
        log_error(f"信号生成异常: {e}")
        return []

# ========================================================
# 【使用示例】
# ========================================================
if __name__ == '__main__':
    # 示例运行：请将日期修改为您想要选股的日期
    TRADE_DATE = '2025-10-14' 
    
    if not os.path.exists(STOCK_DATA_DIR):
        print("\nFATAL: 无法找到数据。请创建 'stock_data' 目录，并放入以股票代码命名的 CSV 文件。")
    else:
        buy_signals = get_daily_buy_signals(TRADE_DATE)
        
        print("\n" + "=" * 50)
        print(f"最终买入信号 ({TRADE_DATE}): {buy_signals}")
        print("=" * 50)
