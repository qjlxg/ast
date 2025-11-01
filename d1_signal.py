# ========================================================
# 项目名称：精简选股信号生成器（不含沪深300指数风控）
# 项目说明：专注于读取本地数据，执行个股选股逻辑，但跳过市场环境检查。
# 目标运行日期：2025-10-31
# ========================================================
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
    "buy_count": 20,              # 最大推荐数量
    "universe2": [],              # 风险股票池（需手动维护）
    "GLOBAL_STOCK_LIST": [],      # 缓存所有A股代码
    # 强制设定为 normal，因为没有指数数据进行检查
    "market_status": "normal"     
}

# ========================================================
# 【日志和辅助函数】 (补充完整)
# ========================================================
def log_info(msg):
    """输出 INFO 级别日志"""
    print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} {msg}")

def log_warning(msg):
    """输出 WARNING 级别日志"""
    print(f"[WARNING] {datetime.now().strftime('%H:%M:%S')} {msg}")

def log_error(msg):
    """输出 ERROR 级别日志"""
    print(f"[ERROR] {datetime.now().strftime('%H:%M:%S')} {msg}")


# ========================================================
# 【数据适配层：文件动态读取】
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
        # 使用 'gbk' 编码读取，适用于国内常见的数据格式
        df = pd.read_csv(file_path, encoding='gbk') 
        df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', 
                           '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        cols_to_keep = ['open', 'high', 'low', 'close', 'volume']
        df = df[[col for col in cols_to_keep if col in df.columns]]
        df = df.astype(float)
        return df
    except Exception as e:
        # log_error(f"读取或清洗文件 {file_path} 失败: {e}")
        return pd.DataFrame()

def mock_get_data(security, end_date, frequency, fields, count):
    """模拟 get_price() 和 get_history()"""
    # 移除后缀以便查找文件
    security_no_suffix = security.split('.')[0]
    file_path = os.path.join(STOCK_DATA_DIR, f"{security_no_suffix}.csv")
    
    df = _read_and_clean_csv(file_path)
    if df.empty: return pd.DataFrame()
        
    end_dt = pd.to_datetime(end_date)
    # 确保日期格式化一致
    filtered_df = df[df.index.normalize() <= end_dt.normalize()]
    
    if count:
        result = filtered_df.tail(count)
    else:
        result = filtered_df
        
    if fields:
        if isinstance(fields, str): fields = [fields]
        # 确保返回的数据框包含请求的字段
        available_fields = [f for f in fields if f in result.columns]
        return result[available_fields]
        
    return result

def mock_get_Ashares():
    """获取全部A股代码列表 (扫描 stock_data 目录)"""
    if g["GLOBAL_STOCK_LIST"]:
        return g["GLOBAL_STOCK_LIST"]
        
    if not os.path.exists(STOCK_DATA_DIR):
        # 不使用 raise，而是返回空列表并在调用处打印 FATAL 错误
        return [] 
        
    files = glob.glob(os.path.join(STOCK_DATA_DIR, '[0-9]*.csv'))
    stock_codes = [os.path.basename(f).replace('.csv', '') for f in files]
    
    final_list = []
    for code in stock_codes:
        if code != INDEX_CODE.split('.')[0]: # 确保排除指数文件
            final_list.append(_get_stock_full_code(code))

    g["GLOBAL_STOCK_LIST"] = final_list
    return final_list

def mock_get_stock_status(all_stocks, status_type, trade_date):
    """模拟获取股票状态 (简化版：仅检查当日是否有数据)"""
    status_map = {}
    dt = pd.to_datetime(trade_date).normalize() # 标准化日期
    
    for stock in all_stocks:
        # 只请求 close 字段，减少数据读取量
        df = mock_get_data(stock, trade_date, '1d', fields=['close'], count=1) 
        
        # 严格检查：如果当日收盘价数据缺失，则视为有风险（停牌、数据缺失等）
        # 检查索引中是否有匹配的日期
        if df.empty or dt not in df.index.normalize(): 
             status_map[stock] = True # True 表示有风险/排除
        else:
            status_map[stock] = False # False 表示无风险/保留
    return status_map

# ========================================================
# 【策略模块】
# ========================================================

def filter_risk_stocks(all_stocks, trade_date):
    """过滤ST/停牌/退市/退市整理期股票 (使用 mock_get_stock_status)"""
    # 这里的逻辑是将所有在 trade_date 缺失数据的股票视为风险股
    st_status = mock_get_stock_status(all_stocks, 'ST', trade_date)
    stocks_to_keep = [stock for stock in all_stocks if not st_status.get(stock)]
    return stocks_to_keep 

def filter_by_volume(stock_list, trade_date, min_vol_ratio=1.2):
    """流动性过滤（成交量）(使用 mock_get_data)"""
    filtered_stocks = []
    for stock in stock_list:
        try:
            # 至少需要25天的量来计算30天平均
            hist_data = mock_get_data(security=stock, end_date=trade_date, frequency='1d', fields=['volume'], count=30)
            if len(hist_data) < 25: continue # 确保有足够的数据
            volumes = hist_data['volume'].values
            
            # 避免索引错误
            recent_avg_vol = np.mean(volumes[-5:])
            long_avg_vol = np.mean(volumes)
            
            # 成交量过滤逻辑：近5日均量 > 过去30日均量 * 1.2
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
            # 只获取当日数据
            kdata = mock_get_data(security=stock, end_date=trade_date, frequency='1d', fields=['open', 'high', 'low', 'close'], count=1)
            if kdata.empty: continue
            
            # 从 Pandas Series 中获取数值
            open_price = kdata['open'].iloc[-1]
            high_price = kdata['high'].iloc[-1]
            low_price = kdata['low'].iloc[-1]
            close_price = kdata['close'].iloc[-1]
            
            body_length = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            lower_shadow = body_bottom - low_price
            
            # 过滤逻辑：长下影线长度 > 实体长度 * 2.0，并且上影线占比不超过 30%
            if body_length > 0 and lower_shadow > body_length * shadow_ratio:
                # 计算上影线长度
                upper_shadow = high_price - max(open_price, close_price)
                
                # 确保 K 线有全长（high-low > 0）
                k_line_full_range = high_price - low_price
                if k_line_full_range > 0 and upper_shadow / k_line_full_range < 0.3:
                    filtered_stocks.append(stock)
            
        except Exception:
            pass # 忽略错误
    return filtered_stocks

def filter_by_rsi(stock_list, trade_date, rsi_period=14, lower_bound=40, upper_bound=65):
    """相对强度(RSI)选股 (使用 mock_get_data)"""
    filtered_stocks = []
    # 至少需要 rsi_period + 1 天的数据才能计算 RSI
    required_count = rsi_period + 5 
    for stock in stock_list:
        try:
            hist_data = mock_get_data(security=stock, end_date=trade_date, frequency='1d', fields=['close'], count=required_count)
            if len(hist_data) < required_count: continue
            
            closes = hist_data['close'].values
            # 使用 talib 计算 RSI
            rsi = ta.RSI(closes, timeperiod=rsi_period)
            current_rsi = rsi[-1]
            
            # 过滤逻辑：RSI 处于 40 到 65 之间
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
        if not all_stocks:
             log_error(f"无法获取A股列表。请检查 '{STOCK_DATA_DIR}' 目录。")
             return []
             
        log_info("原始股票池数量：%d" % len(all_stocks))
        
        # 2. 过滤风险股和手动排除的股票
        safe_stocks = [s for s in all_stocks if s not in g["universe2"]]
        # 这一步会排除在 trade_date 缺失数据的股票
        safe_stocks = filter_risk_stocks(safe_stocks, trade_date)
        log_info("过滤风险股（停牌/数据缺失）后股票数量：%d" % len(safe_stocks))    
        
        # 3. 筛选主板股票 (以60开头或00开头)
        stock_list = [s for s in safe_stocks if s.startswith(('60', '00'))]
        log_info("筛选主板股票后数量：%d" % len(stock_list))
        
        # 4. 核心筛选链：成交量 -> 长下影线 -> RSI
        stock_list = filter_by_volume(stock_list, trade_date, min_vol_ratio=1.2)
        log_info("成交量过滤（近5日均量 > 30日均量*1.2）后股票数量：%d" % len(stock_list))
        
        shadow_list = filter_long_lower_shadow(stock_list, trade_date, shadow_ratio=2.0)
        log_info("长下影线筛选后股票数量：%d" % len(shadow_list))
        
        target_stocks = filter_by_rsi(shadow_list, trade_date, rsi_period=14, lower_bound=40, upper_bound=65)
        log_info("RSI筛选（40-65）后股票数量：%d" % len(target_stocks))
        
        # 5. 备选逻辑
        # 这里的备选逻辑应针对最终结果不足时，使用长下影线池（shadow_list）作为候补
        if len(target_stocks) < g["buy_count"] and shadow_list:
            # 选长下影线池中排名前20的作为备选
            target_stocks.extend(shadow_list[:g["buy_count"]])
            # 去重并截取
            target_stocks = list(set(target_stocks))[:g["buy_count"]]
            log_warning(f"最终筛选股票不足 {g['buy_count']} 只，使用长下影线池的前 {g['buy_count']} 只作为备选。当前数量: {len(target_stocks)}")


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
    # 示例运行：使用您确认的有效数据日期
    TRADE_DATE = '2025-10-31' 
    
    # 运行前的检查
    print("\n" + "#" * 55)
    print("# 项目名称：精简选股信号生成器")
    print("# 目标日期：%s" % TRADE_DATE)
    print("#" * 55)

    if not os.path.exists(STOCK_DATA_DIR):
        print("\nFATAL: 无法找到数据。请创建 'stock_data' 目录，并放入以股票代码命名的 CSV 文件。")
    else:
        buy_signals = get_daily_buy_signals(TRADE_DATE)
        
        print("\n" + "=" * 50)
        print(f"最终买入信号 ({TRADE_DATE}): {buy_signals}")
        print("=" * 50)
