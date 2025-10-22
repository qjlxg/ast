import pandas as pd
import os
import glob
from datetime import datetime
import numpy as np
import warnings

# =========================================================
# 修复 Pandas 警告错误
# 将 pd.core.common.SettingWithCopyWarning 替换为 pd.errors.SettingWithCopyWarning
# =========================================================
try:
    # 尝试使用新版本 Pandas 的路径
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
except AttributeError:
    # 如果失败，尝试使用旧版本 Pandas 的路径 (作为兼容性回退，但通常新版本不包含旧路径)
    try:
        warnings.filterwarnings('ignore', category=pd.core.common.SettingWithCopyWarning)
    except Exception:
        # 实在找不到就忽略，不影响核心功能
        pass 

# --- 配置和路径设置 ---
TODAY_DATE = datetime.now().strftime('%Y%m%d')

# 1. 筛选文件（列表来源）的基础目录
RESULTS_BASE_DIR = 'results' 
# 2. 原始股票历史数据的目录 (例如: stock_data/000001.csv)
STOCK_DATA_DIR = 'stock_data' 

# 3. 最终输出文件的目录
OUTPUT_DIR = f'buy_signals/{TODAY_DATE}'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f'{TODAY_DATE}.csv')

# 原始数据文件必须包含的列名
REQUIRED_RAW_COLUMNS = ['日期', '收盘', '最高', '最低', '股票代码']

# 最终输出 CSV 文件的列名
FINAL_OUTPUT_COLUMNS = [
    '日期', '股票代码', '股票名称', '买入信号总分', '触发信号',
    '收盘价', 'K', 'D', 'J', 'DIF', 'DEA'
]

# 核心计算参数
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
    all_files = glob.glob(os.path.join(base_dir, '**', '*.csv'), recursive=True)
    
    if not all_files:
        print(f"[DEBUG] ⛔️ 在 {base_dir} 及其子目录中未找到任何 CSV 文件。")
        return None
    
    latest_file = max(all_files, key=os.path.getmtime)
    print(f"[DEBUG] ✅ 找到最新的筛选列表文件: {latest_file}")
    return latest_file

# --- 技术指标计算函数 ---
def add_technical_indicators(df):
    """从原始数据计算 MA, MACD, KDJ 指标"""
    
    # 确保关键列为数值类型，并处理缺失值
    for col in ['收盘', '最高', '最低']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['收盘', '最高', '最低']) # 移除价格数据缺失的行

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
    
    # RSV 计算 (处理除以零的情况)
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
def calculate_buy_signal(df_stock, stock_name, code_str):
    """
    基于最新两期数据，计算买入信号得分。
    """
    if len(df_stock) < 2:
        print(f"[DEBUG] ⚠️ {code_str} ({stock_name}): 数据行数不足2行，无法计算信号。")
        return None

    # 获取最新一期（Today）和前一期（Yesterday）数据
    latest = df_stock.iloc[-1]
    prev = df_stock.iloc[-2]

    # 初始化分数和信号
    score = 0
    signals = []

    # C1: KDJ 强势反转 (KDJ_Reversal) - 3分
    # KDJ金叉: K > D 且 K_prev <= D_prev，同时 J 值低于 50 (确保是低位反弹)
    if (latest['K'] > latest['D']) and (prev['K'] <= prev['D']) and (latest['J'] < 50):
        score += SCORE_RULES['KDJ_Reversal']['score']
        signals.append(SCORE_RULES['KDJ_Reversal']['desc'])

    # C2: MACD 金叉 (MACD_GoldenCross) - 3分
    # MACD金叉: DIF > DEA 且 DIF_prev <= DEA_prev
    if (latest['DIF'] > latest['DEA']) and (prev['DIF'] <= prev['DEA']):
        score += SCORE_RULES['MACD_GoldenCross']['score']
        signals.append(SCORE_RULES['MACD_GoldenCross']['desc'])

    # C3: MACD 柱体翻红 (MACD_Bar_Positive) - 2分
    # MACD柱体翻红: MACD_BAR > 0 且 MACD_BAR_prev <= 0
    if (latest['MACD_BAR'] > 0) and (prev['MACD_BAR'] <= 0):
        score += SCORE_RULES['MACD_Bar_Positive']['score']
        signals.append(SCORE_RULES['MACD_Bar_Positive']['desc'])

    # C4: 短期均线确认 (MA_Confirmation) - 2分
    # 短期多头排列: 收盘价 > MA5 且 MA5 > MA20
    if (latest['收盘价'] > latest['MA5']) and (latest['MA5'] > latest['MA20']):
        score += SCORE_RULES['MA_Confirmation']['score']
        signals.append(SCORE_RULES['MA_Confirmation']['desc'])
    
    if score == 0:
        # print(f"[DEBUG] 🚫 {code_str} ({stock_name}): 未触发任何买入信号 (总分 0)。")
        return None

    print(f"[DEBUG] ✨ {code_str} ({stock_name}): 发现信号! 总分 {score}。")
    
    # 构建结果字典
    result = {
        '日期': latest['日期'],
        '股票代码': str(latest['股票代码']).zfill(6),
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
    print(f"预期输出文件: {OUTPUT_FILE}")

    # =========================================================
    # 步骤 1: 读取最新的筛选列表
    # =========================================================
    master_file_path = find_latest_master_file()
    
    if not master_file_path:
        print("错误: 无法继续，未找到筛选列表文件。")
        return

    print(f"-> 正在从筛选文件 {os.path.basename(master_file_path)} 读取待分析股票列表...")
    try:
        # 尝试使用GBK/ANSI和UTF-8两种常见编码读取
        try:
            selected_stocks_df = pd.read_csv(master_file_path, encoding='utf-8')
        except UnicodeDecodeError:
            selected_stocks_df = pd.read_csv(master_file_path, encoding='gbk') 
            
        
        # 检查筛选文件列名，并进行标准化
        col_map = {}
        for col in selected_stocks_df.columns:
            if '股票代码' in col:
                col_map[col] = '股票代码'
            elif '股票名称' in col:
                col_map[col] = '股票名称'

        if '股票代码' not in col_map.values() or '股票名称' not in col_map.values():
            print("错误: 筛选文件缺少 '股票代码' 或 '股票名称' 列，请检查文件格式。")
            return
            
        selected_stocks_df.rename(columns=col_map, inplace=True)
            
        # 将股票代码和名称映射成字典
        selected_stocks_df['股票代码'] = selected_stocks_df['股票代码'].astype(str).str.zfill(6)
        stock_map = selected_stocks_df.set_index('股票代码')['股票名称'].to_dict()
        
    except Exception as e:
        print(f"读取筛选文件 {master_file_path} 时发生严重错误: {e}")
        return

    if not stock_map:
        print("错误: 筛选列表文件为空，没有股票需要分析。")
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
            continue
        
        processed_count += 1
        
        try:
            # 尝试使用 GBK/ANSI 和 UTF-8 两种常见编码读取数据文件
            try:
                df_raw = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df_raw = pd.read_csv(file_path, encoding='gbk')

            # 检查关键列是否存在
            missing_cols = [col for col in REQUIRED_RAW_COLUMNS if col not in df_raw.columns]
            if missing_cols:
                print(f"[DEBUG] ⚠️ 文件 {code_str} 缺少必要原始列: {', '.join(missing_cols)}，跳过。")
                continue
            
            # 排序并计算技术指标
            df_raw = df_raw.sort_values(by='日期', ascending=True).reset_index(drop=True)
            df_calculated = add_technical_indicators(df_raw.copy())

            if df_calculated.empty or len(df_calculated) < MA_LONG:
                 # 确保有足够的数据来计算长周期均线
                 continue

            # 计算信号 
            signal_data = calculate_buy_signal(df_calculated, stock_name, code_str)
            if signal_data:
                all_results.append(signal_data)
            
        except Exception as e:
            print(f"严重警告: 处理文件 {file_path} ({stock_name}) 时发生未预期错误: {e}")
            
    # 转换为 DataFrame
    new_signals_df = pd.DataFrame(all_results)
    
    print(f"-> 实际分析了 {processed_count} 个股票文件。")
    print(f"-> 发现 {len(new_signals_df)} 条新信号需要写入。")

    # =========================================================
    # 步骤 3: 读取旧数据并去重追加
    # =========================================================
    old_signals_df = pd.DataFrame()
    existing_records = set()
    if os.path.exists(OUTPUT_FILE):
        print("-> 发现旧的信号文件，正在读取并进行去重追加...")
        try:
            # 尝试使用 GBK/ANSI 和 UTF-8 两种常见编码读取输出文件
            try:
                old_signals_df = pd.read_csv(OUTPUT_FILE, encoding='utf-8')
            except UnicodeDecodeError:
                old_signals_df = pd.read_csv(OUTPUT_FILE, encoding='gbk')
            
            # 数据标准化
            old_signals_df['股票代码'] = old_signals_df['股票代码'].astype(str).str.zfill(6)
            
            # 记录已有的记录键 (日期, 股票代码)
            existing_records = set(old_signals_df[['日期', '股票代码']].apply(tuple, axis=1))

        except Exception as e:
            print(f"警告: 读取旧文件 {OUTPUT_FILE} 失败: {e}。将跳过旧数据。")
            old_signals_df = pd.DataFrame()

    new_records_to_add = []
    if not new_signals_df.empty:
        for index, row in new_signals_df.iterrows():
            key = (row['日期'], row['股票代码'])
            if key not in existing_records:
                new_records_to_add.append(row.to_dict())
            
    new_signals_df_filtered = pd.DataFrame(new_records_to_add)

    # 合并新旧数据
    if not old_signals_df.empty and not old_signals_df.empty: # 确保旧数据有有效列
        final_df = pd.concat([old_signals_df, new_signals_df_filtered], ignore_index=True, sort=False)
    else:
        final_df = new_signals_df_filtered
        
    # 最终排序和格式化
    if not final_df.empty:
        # 确保列存在且为数值型才能排序
        if '买入信号总分' in final_df.columns:
            final_df['买入信号总分'] = pd.to_numeric(final_df['买入信号总分'], errors='coerce')
            final_df = final_df.sort_values(by=['买入信号总分', '日期'], ascending=[False, True])
        
        final_df = final_df[FINAL_OUTPUT_COLUMNS] # 确保列顺序

    # =========================================================
    # 步骤 4: 输出结果
    # =========================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if final_df.empty:
        print("--- 任务完成 ---")
        print("-> 🚫 最终结果集为空，未写入任何文件。")
    else:
        # 统一使用 UTF-8 编码写入
        final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        print(f"--- 任务完成 ---")
        print(f"✅ 已将 {len(new_records_to_add)} 条新信号追加到 {OUTPUT_FILE} 中。")
        print(f"✅ 最终结果包含 {len(final_df)} 条记录（已去重）。")

if __name__ == '__main__':
    main()
