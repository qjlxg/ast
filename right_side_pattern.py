import os
import pandas as pd
from datetime import datetime, timedelta
import pytz
from multiprocessing import Pool, cpu_count
import numpy as np
from typing import List, Optional, Dict
import logging

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- 配置 ---
INPUT_DIR = "stock_data"              # 股票历史数据目录
BUY_SIGNALS_BASE_DIR = "buy_signals"  # 信号文件基础目录（含完整CSV）
OUTPUT_DIR = "results"                # 输出目录
TIMEZONE = "Asia/Shanghai"
NUM_PROCESSES = max(1, cpu_count() - 1)

# --- 技术指标计算 ---
def calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
    df['EMA12'] = df['收盘'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['收盘'].ewm(span=26, adjust=False).mean()
    df['DIF'] = df['EMA12'] - df['EMA26']
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    return df

def calculate_kdj(df: pd.DataFrame, n=9, k_p=3, d_p=3) -> pd.DataFrame:
    low_n = df['最低'].rolling(n, min_periods=1).min()
    high_n = df['最高'].rolling(n, min_periods=1).max()
    rsv = (df['收盘'] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    rsv = rsv.fillna(50)

    df['K'] = rsv.ewm(alpha=1/k_p, adjust=False).mean()
    df['D'] = df['K'].ewm(alpha=1/d_p, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    return df

# --- 数据预处理 ---
def preprocess_stock(df: pd.DataFrame) -> pd.DataFrame:
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
    df = df.dropna(subset=['日期']).sort_values('日期').reset_index(drop=True)
    cols = ['开盘', '收盘', '最高', '最低', '成交量']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=cols)

# --- 核心筛选逻辑 ---
def process_stock_file(stock_code: str) -> Optional[Dict]:
    filepath = os.path.join(INPUT_DIR, f"{stock_code}.csv")
    if not os.path.exists(filepath):
        logger.debug(f"股票数据文件不存在: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath)
        if len(df) < 30:
            return None

        df = preprocess_stock(df)
        df = calculate_macd(df)
        df = calculate_kdj(df)
        df['20D_Max'] = df['收盘'].rolling(20, min_periods=20).max()

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # 实盘过滤
        if latest['成交量'] <= 100 or prev['收盘'] <= 0:
            return None
        daily_change = latest['收盘'] / prev['收盘'] - 1
        if abs(daily_change) >= 0.095:
            return None
        if pd.isna(df['20D_Max'].iloc[-2]):
            return None

        # 右侧加仓条件
        macd_bullish = (latest['DIF'] > 0.1) and (latest['DEA'] > 0.1) and (latest['DIF'] > latest['DEA'])
        macd_rising = (latest['DIF'] > prev['DIF']) and (latest['DEA'] > prev['DEA'])
        j_dead_cross = (prev['J'] > prev['K']) and (latest['J'] <= latest['K']) and (latest['J'] < 90)
        j_drop = (prev['J'] - latest['J']) / prev['J'] if prev['J'] > 0 else 0
        volume_rising = latest['成交量'] > prev['成交量'] * 1.2
        not_new_high = latest['收盘'] < df['20D_Max'].iloc[-2]

        if all([macd_bullish, macd_rising, j_dead_cross, j_drop > 0.05, volume_rising, not_new_high]):
            return {
                '股票代码': stock_code,
                '日期': latest['日期'].strftime('%Y-%m-%d'),
                '收盘': round(latest['收盘'], 2),
                '涨跌幅': round(daily_change * 100, 2),
                'J值跌幅%': round(j_drop * 100, 1),
                'MACD_DIF': round(latest['DIF'], 4),
                'MACD_DEA': round(latest['DEA'], 4),
                'KDJ_J': round(latest['J'], 1),
                '说明': '右侧加仓点：MACD多头抬升，KDJ J值显著死叉回调'
            }
    except Exception as e:
        logger.error(f"处理 {stock_code} 失败: {e}")
    return None

# --- 查找并读取完整信号文件 ---
def find_signal_file() -> List[str]:
    """动态查找最近三天内的信号文件 buy_signals/YYYYMM/YYYYMMDD.csv"""
    tz = pytz.timezone(TIMEZONE)
    today = datetime.now(tz).date()
    logger.info(f"当前上海日期: {today.strftime('%Y-%m-%d')}，开始动态搜索信号文件...")

    for delta in range(3):  # 今天(0)、昨天(1)、前天(2)
        date = today - timedelta(days=delta)
        date_str = date.strftime("%Y%m%d")  # YYYYMMDD
        ym_dir = date.strftime("%Y%m")      # YYYYMM
        signal_path = os.path.join(BUY_SIGNALS_BASE_DIR, ym_dir, f"{date_str}.csv")
        logger.debug(f"检查路径: {signal_path}")

        if os.path.exists(signal_path):
            logger.info(f"找到信号文件: {signal_path}")
            try:
                df = pd.read_csv(signal_path)
                required_cols = ['股票代码']
                if all(col in df.columns for col in required_cols):
                    # 保留最新日期的记录（避免重复）
                    df['日期'] = pd.to_datetime(df['日期'])
                    df = df.sort_values('日期').drop_duplicates('股票代码', keep='last')
                    codes = df['股票代码'].astype(str).str.zfill(6).tolist()
                    logger.info(f"提取到 {len(codes)} 只独特股票代码")
                    return codes
                else:
                    logger.error(f"信号文件缺少必要列: {required_cols}")
            except Exception as e:
                logger.error(f"读取失败 {signal_path}: {e}")
    logger.warning("未找到最近三天的信号文件")
    return []

# --- 主函数 ---
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stock_codes = find_signal_file()
    if not stock_codes:
        logger.info("无信号文件或无股票代码，程序结束")
        return

    if not os.path.exists(INPUT_DIR):
        logger.error(f"数据目录不存在: {INPUT_DIR}")
        return

    logger.info(f"开始筛选 {len(stock_codes)} 只股票的右侧加仓信号...")

    with Pool(NUM_PROCESSES) as pool:
        results = pool.map(process_stock_file, stock_codes)

    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        logger.info("未发现符合右侧加仓条件的股票")
        return

    # 保存结果
    tz = pytz.timezone(TIMEZONE)
    now = datetime.now(tz)
    ym = now.strftime("%Y%m")
    ts = now.strftime("%Y%m%d%H%M%S")
    out_dir = os.path.join(OUTPUT_DIR, ym)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"right_side_pattern_enhanced_{ts}.csv")

    pd.DataFrame(valid_results).to_csv(out_path, index=False, encoding='utf-8-sig')
    logger.info(f"成功输出 {len(valid_results)} 条右侧加仓信号 → {out_path}")

if __name__ == "__main__":
    main()
