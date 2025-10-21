import pandas as pd
import os
import glob
from datetime import datetime
import csv
import numpy as np
import warnings

# 忽略SettingWithCopyWarning，因为我们知道自己在做什么
warnings.filterwarnings('ignore', category=pd.core.common.SettingWithCopyWarning)

# --- 配置和路径设置 ---
TODAY_DATE = datetime.now().strftime('%Y%m%d')

# 1. 筛选文件（列表来源）的基础目录
RESULTS_BASE_DIR = 'results' 
# 2. 原始股票历史数据的目录 (例如: stock_data/000001.csv)
STOCK_DATA_DIR = 'stock_data' 

# 3. 最终输出文件的目录
OUTPUT_DIR = f'buy_signals/{TODAY_DATE}'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f'{TODAY_DATE}.csv')

# 原始数据文件必须包含的列名 (基于您上传的CSV格式)
REQUIRED_RAW_COLUMNS = ['日期', '收盘', '最高', '最低', '股票代码']

# 最终输出 CSV 文件的列名
FINAL_OUTPUT_COLUMNS = [
    '日期', '股票代码', '股票名称', '买入信号总分', '触发信号',
    '收盘价', 'K', 'D', 'J', 'DIF', 'DEA'
]

# 核心计算参数 (与上个版本保持一致)
MA_SHORT = 5
MA_LONG = 20
MACD_SHORT = 12
MACD_LONG = 26
MACD_SIGNAL = 9
KDJ_N = 9
KDJ_M1 = 3
KDJ_M2 = 3

# --- 辅助函数：查找最新的筛选列表文件 ---
def find_latest_master_file(base_dir=RESULTS_BASE_DIR):
    """查找 'results' 目录下所有子目录中最新修改的 CSV 文件作为分析列表"""
    # 递归查找 base_dir 及其所有子目录中的所有 CSV 文件
    all_files = glob.glob(os.path.join(base_dir, '**', '*.csv'), recursive=True)
    
    if not all_files:
        return None
    
    # 找到最新修改的文件
    latest_file = max(all_files, key=os.path.getmtime)
    return latest_file

# --- 技术指标计算函数 (与上个版本保持一致) ---
def add_technical_indicators(df):
    """从原始数据计算 MA, MACD, KDJ 指标"""
    
    # 1. 均线 (MA)
    df['MA5'] = df['收盘'].rolling(window=MA_SHORT).mean()
    df['MA20'] = df['收盘'].rolling(window=MA_LONG).mean()

    # 2. MACD
    ema_short = df['收盘'].ewm(span=MACD_SHORT, adjust=False).mean()
    ema_long = df['收盘'].ewm(span=MACD_LONG, adjust=False).mean()
    df['DIF'] = ema_short - ema_long
    df['DEA'] = df['DIF'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df['MACD_BAR'] = 2 * (df['DIF'] - df['DEA'])
    
    # 3. KDJ (RSV, K, D, J)
    df['LLV'] = df['最低'].rolling(window=KDJ_N).min()
    df['HHV'] = df['最高'].rolling(window=KDJ_N).max()
    denominator = df['HHV'] - df['LLV']
    df['RSV'] = np.where(denominator == 0, 
                         50.0, 
                         (df['收盘'] - df['LLV']) / denominator * 100)
    df['K'] = df['RSV'].ewm(span=KDJ_M1, adjust=False).mean()
    df['D'] = df['K'].ewm(span=KDJ_M2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    df = df.drop(columns=['LLV', 'HHV', 'RSV'], errors='ignore')

    return df.dropna().rename(columns={'收盘': '收盘价'})

# --- 评分规则 ---
SCORE_RULES = {
    'KDJ_Reversal': {'score': 3, 'desc': 'KDJ低位反转金叉'},
    'MACD_GoldenCross': {'score': 3, 'desc': 'MACD金叉'},
    'MACD_Bar_Positive': {'score': 2, 'desc': 'MACD柱体由绿转红'},
    'MA_Confirmation': {'score': 2, 'desc': '短期均线多头确认'}
}

# --- 核心分析函数 ---
def calculate_buy_signal(df_stock, stock_name):
    """
    基于最新两期数据，计算买入信号得分。
    """
    if len(df_stock) < 2:
        return None

    # 获取最新一期（Today）和前一期（Yesterday）数据
    latest = df_stock.iloc[-1]
    prev = df_stock.iloc[-2]

    # 初始化分数和信号
    score = 0
    signals = []

    # C1: KDJ 强势反转 (KDJ_Reversal) - 3分
    if (latest['K'] > latest['D']) and (prev['K'] <= prev['D']) and (latest['J'] < 50):
        score += SCORE_RULES['KDJ_Reversal']['score']
        signals.append(SCORE_RULES['KDJ_Reversal']['desc'])

    # C2: MACD 金叉 (MACD_GoldenCross) - 3分
    if (latest['DIF'] > latest['DEA']) and (prev['DIF'] <= prev['DEA']):
        score += SCORE_RULES['MACD_GoldenCross']['score']
        signals.append(SCORE_RULES['MACD_GoldenCross']['desc'])

    # C3: MACD 柱体翻红 (MACD_Bar_Positive) - 2分
    if (latest['MACD_BAR'] > 0) and (prev['MACD_BAR'] <= 0):
        score += SCORE_RULES['MACD_Bar_Positive']['score']
        signals.append(SCORE_RULES['MACD_Bar_Positive']['desc'])

    # C4: 短期均线确认 (MA_Confirmation) - 2分
    if (latest['收盘价'] > latest['MA5']) and (latest['MA5'] > latest['MA20']):
        score += SCORE_RULES['MA_Confirmation']['score']
        signals.append(SCORE_RULES['MA_Confirmation']['desc'])
    
    if score == 0:
        return None

    stock_code = str(latest['股票代码']).zfill(6)
    
    # 构建结果字典
    result = {
        '日期': latest['日期'],
        '股票代码': stock_code,
        '股票名称': stock_name, # 使用从筛选列表获取的名称
        '买入信号总分': score,
        '触发信号': '，'.join(signals),
        '收盘价': latest['收盘价'],
        'K': round(latest['K'], 2),
        'D': round(latest['D'], 2),
        'J': round(latest['J'], 2),
        'DIF': round(latest['DIF'], 3),
        'DEA': round(latest['DEA'], 3),
    }
    return result

# --- 主执行流程 ---
def main():
    """主程序，负责文件读取、计算指标、分析和结果输出"""
    print(f"--- 股票定向买入信号分析程序 ---")
    print(f"分析日期: {TODAY_DATE}")
    print(f"数据目录: {STOCK_DATA_DIR}")
    print(f"输出文件: {OUTPUT_FILE}")

    # =========================================================
    # 步骤 1: 读取最新的筛选列表 (定向分析的关键)
    # =========================================================
    master_file_path = find_latest_master_file()
    
    if not master_file_path:
        print(f"错误: 找不到 {RESULTS_BASE_DIR} 目录下任何 CSV 文件作为待分析列表。请确保您已生成筛选列表。")
        return

    print(f"-> 正在从筛选文件 {os.path.basename(master_file_path)} 读取待分析股票列表...")
    try:
        selected_stocks_df = pd.read_csv(master_file_path, encoding='utf-8')
        
        # 将股票代码和名称映射成字典
        selected_stocks_df['股票代码'] = selected_stocks_df['股票代码'].apply(lambda x: str(x).zfill(6))
        stock_map = selected_stocks_df.set_index('股票代码')['股票名称'].to_dict()
        
    except KeyError:
        print("错误: 筛选文件缺少 '股票代码' 或 '股票名称' 列，请检查文件格式。")
        return
    except Exception as e:
        print(f"读取筛选文件 {master_file_path} 时发生错误: {e}")
        return

    print(f"-> 共找到 {len(stock_map)} 支股票需要进行分析。")
    all_results = []
    processed_count = 0
    
    # =========================================================
    # 步骤 2: 定向加载和分析历史数据
    # =========================================================
    for code_str, stock_name in stock_map.items():
        file_path = os.path.join(STOCK_DATA_DIR, f'{code_str}.csv')
        
        if not os.path.exists(file_path):
            # print(f"警告: 原始数据文件 {file_path} 不存在，跳过 {stock_name}。")
            continue
        
        processed_count += 1
        
        try:
            df_raw = pd.read_csv(file_path, encoding='utf-8')
            
            # 检查关键列是否存在
            missing_cols = [col for col in REQUIRED_RAW_COLUMNS if col not in df_raw.columns]
            if missing_cols:
                print(f"警告: 文件 {os.path.basename(file_path)} 缺少必要原始列: {', '.join(missing_cols)}，跳过。")
                continue
            
            # 排序并计算技术指标
            df_raw = df_raw.sort_values(by='日期', ascending=True).reset_index(drop=True)
            df_calculated = add_technical_indicators(df_raw.copy())

            if df_calculated.empty or len(df_calculated) < 2:
                 # print(f"警告: 文件 {os.path.basename(file_path)} 数据不足或指标计算失败，跳过。")
                 continue

            # 计算信号 (传入股票名称)
            signal_data = calculate_buy_signal(df_calculated, stock_name)
            if signal_data:
                all_results.append(signal_data)
            
        except Exception as e:
            print(f"处理文件 {file_path} ({stock_name}) 时发生错误: {e}")
            
    # 转换为 DataFrame
    new_signals_df = pd.DataFrame(all_results)
    
    print(f"-> 实际分析了 {processed_count} 个股票文件。")
    if new_signals_df.empty:
        print("-> 未发现任何符合条件的买入信号。")

    # =========================================================
    # 步骤 3: 读取旧数据并去重追加 (与上个版本保持一致)
    # =========================================================
    old_signals_df = pd.DataFrame()
    existing_records = set()
    if os.path.exists(OUTPUT_FILE):
        print("-> 发现旧的信号文件，正在读取并进行去重追加...")
        try:
            # 读取旧数据并记录已有的 (日期, 股票代码) 组合
            old_signals_df = pd.read_csv(OUTPUT_FILE, encoding='utf-8')
            old_signals_df['股票代码'] = old_signals_df['股票代码'].apply(lambda x: str(x).zfill(6))
            
            # 记录已有的记录键 (日期, 股票代码)
            existing_records = set(old_signals_df[['日期', '股票代码']].apply(tuple, axis=1))

        except Exception as e:
            print(f"读取旧文件 {OUTPUT_FILE} 失败: {e}。将跳过旧数据。")
            old_signals_df = pd.DataFrame()

    # 过滤掉新数据中已存在的旧记录
    new_records_to_add = []
    if not new_signals_df.empty:
        for index, row in new_signals_df.iterrows():
            key = (row['日期'], row['股票代码'])
            if key not in existing_records:
                new_records_to_add.append(row.to_dict())
            
    new_signals_df_filtered = pd.DataFrame(new_records_to_add)

    # 合并新旧数据
    if not old_signals_df.empty:
        final_df = pd.concat([old_signals_df, new_signals_df_filtered], ignore_index=True)
    else:
        final_df = new_signals_df_filtered
        
    # 最终排序和格式化
    if not final_df.empty:
        final_df['买入信号总分'] = pd.to_numeric(final_df['买入信号总分'])
        final_df = final_df.sort_values(by=['买入信号总分', '日期'], ascending=[False, True])
        final_df = final_df[FINAL_OUTPUT_COLUMNS] # 确保列顺序

    # 4. 输出结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if final_df.empty:
        print("警告: 最终无记录写入。")
    else:
        final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        print(f"--- 任务完成 ---")
        print(f"已将 {len(new_records_to_add)} 条新信号追加到 {OUTPUT_FILE} 中。")
        print(f"最终结果包含 {len(final_df)} 条记录（已去重）。")

if __name__ == '__main__':
    main()
