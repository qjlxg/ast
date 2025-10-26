# -*- coding: utf-8 -*-
import os
import pandas as pd
from datetime import datetime, timedelta
import pytz
from multiprocessing import Pool, cpu_count
import numpy as np
from typing import List, Optional, Dict
import logging

# ---------- 日志 ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# ---------- 配置 ----------
INPUT_DIR = "stock_data" # 历史日线
BUY_SIGNALS_BASE_DIR = "buy_signals" # 信号文件根目录
OUTPUT_DIR = "results"
TIMEZONE = "Asia/Shanghai"
NUM_PROCESSES = max(1, cpu_count() - 1)
STOCK_CODE_COLUMN_NAME = '股票代码' # <--- 明确信号文件中的股票代码列名

# ---------- 指标 ----------
def calc_macd(df: pd.DataFrame) -> pd.DataFrame:
    """计算 MACD 指标 (DIF, DEA)"""
    df['EMA12'] = df['收盘'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['收盘'].ewm(span=26, adjust=False).mean()
    df['DIF'] = df['EMA12'] - df['EMA26']
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    return df

def calc_kdj(df: pd.DataFrame, n=9, k_p=3, d_p=3) -> pd.DataFrame:
    """计算 KDJ 指标 (K, D, J)"""
    low = df['最低'].rolling(n, min_periods=1).min()
    high = df['最高'].rolling(n, min_periods=1).max()
    # 避免除以零，并将初始值设为 50
    range_diff = (high - low).replace(0, np.nan)
    rsv = (df['收盘'] - low) / range_diff * 100
    rsv = rsv.fillna(50)

    # 使用指数平滑移动平均线 (EMA) 计算 K 和 D
    df['K'] = rsv.ewm(alpha=1/k_p, adjust=False).mean()
    df['D'] = df['K'].ewm(alpha=1/d_p, adjust=False).mean()
    df['J'] = 3*df['K'] - 2*df['D']
    return df

# ---------- 数据清洗 ----------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """清理并格式化历史数据"""
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
    df = df.dropna(subset=['日期']).sort_values('日期').reset_index(drop=True)
    for c in ['开盘','收盘','最高','最低','成交量']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # 确保价格和成交量数据完整
    return df.dropna(subset=['开盘','收盘','最高','最低','成交量'])

# ---------- 单只股票筛选 ----------
def process(stock_code: str) -> Optional[Dict]:
    """处理单个股票数据，进行模式筛选"""
    fp = os.path.join(INPUT_DIR, f"{stock_code}.csv")
    if not os.path.exists(fp):
        # log.debug(f"跳过：历史文件 {fp} 不存在")
        return None

    try:
        df = pd.read_csv(fp)
        
        # 数据预处理
        df = preprocess(df)
        if len(df) < 30: return None # 至少需要 30 天数据
        
        # 指标计算
        df = calc_macd(df)
        df = calc_kdj(df)
        # 滚动计算 20 日收盘价最大值 (至少需要 20 天数据)
        df['20D_Max'] = df['收盘'].rolling(20, min_periods=20).max()

        # 检查数据长度 (至少需要 2 天用于比较)
        if len(df) < 2: return None

        cur = df.iloc[-1]
        pre = df.iloc[-2]

        # 1. 实盘过滤 (停牌、涨跌停、数据缺失)
        if cur['成交量'] <= 100 or pre['收盘'] <= 0: return None # 过滤极低成交量和无效价格
        chg = cur['收盘']/pre['收盘'] - 1
        if abs(chg) >= 0.095: return None # 过滤涨跌停
        
        # 检查 20D_Max 是否已计算
        if pd.isna(df['20D_Max'].iloc[-2]): return None

        # 2. 右侧加仓条件
        
        # C. MACD 强多头形态
        macd_ok = (cur['DIF'] > 0.1) and (cur['DEA'] > 0.1) and (cur['DIF'] > cur['DEA'])
        
        # D. MACD 趋势向上
        macd_up = (cur['DIF'] > pre['DIF']) and (cur['DEA'] > pre['DEA'])
        
        # E. KDJ J 值死叉形态 (J从K上方跌至K下方，且J<90)
        j_cross = pre['J'] > pre['K'] and cur['J'] <= cur['K'] and cur['J'] < 90
        
        # F. J 值显著回落 (J值跌幅超过 5%)
        j_drop = (pre['J'] - cur['J']) / pre['J'] if pre['J'] > 0 else 0
        j_drop_significant = j_drop > 0.05
        
        # G. 成交量放量 (最新成交量比前一天增加 20% 以上)
        vol_up = cur['成交量'] > pre['成交量'] * 1.2
        
        # H. 非新高 (最新收盘价低于前一天计算的 20 日收盘价最高点)
        not_high = cur['收盘'] < df['20D_Max'].iloc[-2]

        # 3. 组合筛选
        if all([macd_ok, macd_up, j_cross, j_drop_significant, vol_up, not_high]):
            return {
                '股票代码': stock_code,
                '日期': cur['日期'].strftime('%Y-%m-%d'),
                '收盘': round(cur['收盘'], 2),
                '涨跌幅': round(chg*100, 2),
                'J值跌幅%': round(j_drop*100, 1),
                'MACD_DIF': round(cur['DIF'], 4),
                'MACD_DEA': round(cur['DEA'], 4),
                'KDJ_J': round(cur['J'], 1),
                '说明': '右侧加仓点：MACD多头抬升，KDJ J值显著死叉回调'
            }
            
    except Exception as e:
        log.error(f"处理 {stock_code} 出错: {e}")
    return None

# ---------- 动态查找信号文件 ----------
def find_signal_file() -> List[str]:
    """
    尝试从 buy_signals 目录读取今天的、昨天的、或前天的信号文件。
    支持 buy_signals/YYYYMMDD/YYYYMMDD.csv (用户要求) 和 buy_signals/YYYYMM/YYYYMMDD.csv 两种结构。
    """
    tz = pytz.timezone(TIMEZONE)
    now = datetime.now(tz).date()
    log.info(f"当前上海日期: {now}，开始搜索信号文件...")

    codes = []
    # 搜索范围：今天, 昨天, 前天
    for d in range(3):
        date = now - timedelta(days=d)
        ym = date.strftime("%Y%m") # 202510
        dd = date.strftime("%Y%m%d") # 20251024

        # 路径方案 1：按天 (用户要求: buy_signals/YYYYMMDD/YYYYMMDD.csv)
        path1 = os.path.join(BUY_SIGNALS_BASE_DIR, dd, f"{dd}.csv")
        # 路径方案 2：按月 (兼容性: buy_signals/YYYYMM/YYYYMMDD.csv)
        path2 = os.path.join(BUY_SIGNALS_BASE_DIR, ym, f"{dd}.csv")

        for path, scheme in [(path1, "按天"), (path2, "按月")]:
            if not os.path.exists(path):
                # log.debug(f"  检查 [{date}] [{scheme}] → {path} 不存在")
                continue

            log.info(f"  成功找到信号文件 [{date}] [{scheme}] → {path}")

            # 读取文件
            try:
                # dtype=str 确保读取的代码是字符串，避免数值格式化问题
                df = pd.read_csv(path, dtype={STOCK_CODE_COLUMN_NAME: str}) 
                
                if STOCK_CODE_COLUMN_NAME not in df.columns:
                    log.error(f"  文件缺少配置的列名 '{STOCK_CODE_COLUMN_NAME}': {path}")
                    continue

                # 数据处理：去重，只保留最新的记录
                df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
                df = df.sort_values('日期').drop_duplicates(STOCK_CODE_COLUMN_NAME, keep='last')
                
                # 格式化股票代码为 6 位字符串 (例如 000001)
                cur_codes = df[STOCK_CODE_COLUMN_NAME].astype(str).str.zfill(6).tolist()
                
                # 过滤无效代码
                valid_codes = [code for code in cur_codes if len(code) == 6 and code.isdigit()]

                codes.extend(valid_codes)
                log.info(f"  从 {path} 读取到 {len(valid_codes)} 只有效股票")
                
                # 找到即返回，优先最新日期的文件
                # list(dict.fromkeys(codes)) 保持原始顺序进行去重
                return list(dict.fromkeys(codes))
                
            except Exception as e:
                log.error(f"  读取 {path} 失败: {e}")

    log.warning(f"未在最近 3 天内找到任何信号文件（{BUY_SIGNALS_BASE_DIR} 目录）")
    return []

# ---------- 主流程 ----------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stock_codes = find_signal_file()
    if not stock_codes:
        log.info("没有可处理的股票代码，程序结束")
        return

    if not os.path.isdir(INPUT_DIR):
        log.error(f"历史数据目录不存在: {INPUT_DIR}")
        return

    log.info(f"准备并行处理 {len(stock_codes)} 只股票（{NUM_PROCESSES} 进程）")
    with Pool(NUM_PROCESSES) as pool:
        # 使用 pool.map 并行处理
        results = pool.map(process, stock_codes)

    # 过滤空结果
    valid = [r for r in results if r]
    if not valid:
        log.info("没有符合右侧加仓条件的股票")
        return

    # 结果输出
    tz = pytz.timezone(TIMEZONE)
    now = datetime.now(tz)
    out_dir = os.path.join(OUTPUT_DIR, now.strftime("%Y%m"))
    os.makedirs(out_dir, exist_ok=True)
    
    # 构造输出文件名 (包含时间戳)
    out_path = os.path.join(out_dir,
                            f"right_side_pattern_enhanced_{now.strftime('%Y%m%d%H%M%S')}.csv")

    # 使用 utf-8-sig 编码以确保 Excel 中文显示无乱码
    pd.DataFrame(valid).to_csv(out_path, index=False, encoding='utf-8-sig')
    log.info(f"成功输出 {len(valid)} 条记录 → {out_path}")

if __name__ == "__main__":
    # 强制使用上海时区 (Linux/Mac)
    os.environ['TZ'] = TIMEZONE
    try:
        import time
        time.tzset()
    except AttributeError: # Windows 不支持 os.environ['TZ']
        pass
    main()
