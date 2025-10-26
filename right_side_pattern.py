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
INPUT_DIR            = "stock_data"               # 历史日线
BUY_SIGNALS_BASE_DIR = "buy_signals"               # 信号文件根目录
OUTPUT_DIR           = "results"
TIMEZONE             = "Asia/Shanghai"
NUM_PROCESSES        = max(1, cpu_count() - 1)

# ---------- 指标 ----------
def calc_macd(df: pd.DataFrame) -> pd.DataFrame:
    df['EMA12'] = df['收盘'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['收盘'].ewm(span=26, adjust=False).mean()
    df['DIF']   = df['EMA12'] - df['EMA26']
    df['DEA']   = df['DIF'].ewm(span=9, adjust=False).mean()
    return df

def calc_kdj(df: pd.DataFrame, n=9, k_p=3, d_p=3) -> pd.DataFrame:
    low  = df['最低'].rolling(n, min_periods=1).min()
    high = df['最高'].rolling(n, min_periods=1).max()
    rsv  = (df['收盘'] - low) / (high - low).replace(0, np.nan) * 100
    rsv  = rsv.fillna(50)

    df['K'] = rsv.ewm(alpha=1/k_p, adjust=False).mean()
    df['D'] = df['K'].ewm(alpha=1/d_p, adjust=False).mean()
    df['J'] = 3*df['K'] - 2*df['D']
    return df

# ---------- 数据清洗 ----------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
    df = df.dropna(subset=['日期']).sort_values('日期').reset_index(drop=True)
    for c in ['开盘','收盘','最高','最低','成交量']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=['开盘','收盘','最高','最低','成交量'])

# ---------- 单只股票筛选 ----------
def process(stock_code: str) -> Optional[Dict]:
    fp = os.path.join(INPUT_DIR, f"{stock_code}.csv")
    if not os.path.exists(fp):
        return None

    try:
        df = pd.read_csv(fp)
        if len(df) < 30: return None
        df = preprocess(df)
        df = calc_macd(df)
        df = calc_kdj(df)
        df['20D_Max'] = df['收盘'].rolling(20, min_periods=20).max()

        cur = df.iloc[-1]
        pre = df.iloc[-2]

        # 实盘过滤
        if cur['成交量'] <= 100 or pre['收盘'] <= 0: return None
        chg = cur['收盘']/pre['收盘'] - 1
        if abs(chg) >= 0.095: return None
        if pd.isna(df['20D_Max'].iloc[-2]): return None

        # 右侧加仓条件
        macd_ok   = cur['DIF']>0.1 and cur['DEA']>0.1 and cur['DIF']>cur['DEA']
        macd_up   = cur['DIF']>pre['DIF'] and cur['DEA']>pre['DEA']
        j_cross   = pre['J']>pre['K'] and cur['J']<=cur['K'] and cur['J']<90
        j_drop    = (pre['J']-cur['J'])/pre['J'] if pre['J']>0 else 0
        vol_up    = cur['成交量'] > pre['成交量']*1.2
        not_high  = cur['收盘'] < df['20D_Max'].iloc[-2]

        if macd_ok and macd_up and j_cross and j_drop>0.05 and vol_up and not_high:
            return {
                '股票代码': stock_code,
                '日期'    : cur['日期'].strftime('%Y-%m-%d'),
                '收盘'    : round(cur['收盘'],2),
                '涨跌幅'  : round(chg*100,2),
                'J值跌幅%': round(j_drop*100,1),
                'MACD_DIF': round(cur['DIF'],4),
                'MACD_DEA': round(cur['DEA'],4),
                'KDJ_J'   : round(cur['J'],1),
                '说明'    : '右侧加仓点：MACD多头抬升，KDJ J值显著死叉回调'
            }
    except Exception as e:
        log.error(f"处理 {stock_code} 出错: {e}")
    return None

# ---------- 动态查找信号文件 ----------
def find_signal_file() -> List[str]:
    tz   = pytz.timezone(TIMEZONE)
    now  = datetime.now(tz).date()
    log.info(f"当前上海日期: {now}，开始搜索信号文件（最多向前 3 天）")

    codes = []
    for d in range(3):
        date = now - timedelta(days=d)
        ym   = date.strftime("%Y%m")       # 202510
        dd   = date.strftime("%Y%m%d")     # 20251024
        path = os.path.join(BUY_SIGNALS_BASE_DIR, ym, f"{dd}.csv")

        # ---- 详细诊断信息 ----
        exists = os.path.exists(path)
        log.info(f"  检查 [{d}] {date} → {path}  {'存在' if exists else '不存在'}")

        if not exists:
            # 给出常见原因
            parent = os.path.dirname(path)
            if not os.path.isdir(parent):
                log.warning(f"  父目录不存在: {parent}")
            continue

        # ---- 读取文件 ----
        try:
            df = pd.read_csv(path, dtype=str)   # 先全字符，防止自动转换
            if '股票代码' not in df.columns:
                log.error(f"  文件缺少 '股票代码' 列: {path}")
                continue

            df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
            df = df.sort_values('日期').drop_duplicates('股票代码', keep='last')
            cur_codes = df['股票代码'].astype(str).str.zfill(6).tolist()
            codes.extend(cur_codes)
            log.info(f"  从 {path} 读取到 {len(cur_codes)} 只股票（去重后）")
            return list(dict.fromkeys(codes))   # 整体去重并保持首次出现顺序
        except Exception as e:
            log.error(f"  读取 {path} 失败: {e}")

    log.warning("未在最近 3 天内找到任何信号文件")
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
        results = pool.map(process, stock_codes)

    valid = [r for r in results if r]
    if not valid:
        log.info("没有符合右侧加仓条件的股票")
        return

    tz = pytz.timezone(TIMEZONE)
    now = datetime.now(tz)
    out_dir = os.path.join(OUTPUT_DIR, now.strftime("%Y%m"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir,
                f"right_side_pattern_enhanced_{now.strftime('%Y%m%d%H%M%S')}.csv")

    pd.DataFrame(valid).to_csv(out_path, index=False, encoding='utf-8-sig')
    log.info(f"成功输出 {len(valid)} 条记录 → {out_path}")

if __name__ == "__main__":
    # 强制使用上海时区（配合你手动 export TZ='Asia/Shanghai'）
    os.environ['TZ'] = TIMEZONE
    try:
        import time
        time.tzset()
    except AttributeError:   # Windows 不支持 tzset
        pass
    main()
