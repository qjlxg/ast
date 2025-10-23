import pandas as pd
import glob
import os
from datetime import datetime
import pytz

def combine_and_sort_csvs():
    """
    读取 results/YYYYMMDD/ 下的所有 CSV 文件，与已存在的合并文件追加、去重、排序，
    并保存到 combined_results/YYYYMMDD/combined_buy_signals.csv。
    
    该脚本严格遵循用户提供的格式要求：
    1. 确保 '股票代码' 列作为字符串处理，保留前导零。
    2. 合并所有文件和历史结果。
    3. 去重，保留评分最高的记录。
    4. 按评分降序排序。
    5. 输出的CSV文件不包含索引。
    """
    # 设定时区为上海，并获取当前日期作为目录名 (YYYYMMDD)
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    now_shanghai = datetime.now(shanghai_tz)
    date_str = now_shanghai.strftime('%Y%m%d')
    
    # 构造输入目录路径
    input_dir_pattern = f'results/{date_str}/*.csv'
    
    # 构造输出目录路径和文件名
    output_dir = f'combined_results/{date_str}'
    output_file = os.path.join(output_dir, 'combined_buy_signals.csv')
    
    # 定义需要的列名 (严格依据用户提供的源文件格式)
    required_columns = ['股票代码', '股票名称', '买入信号', '评分', '图表路径']
    
    # 核心：定义数据类型，确保 '股票代码' 为字符串以保留前导零
    dtype_spec = {'股票代码': str}

    # --- 1. 读取当天所有新的 CSV 文件 ---
    all_new_files = glob.glob(input_dir_pattern)
    
    if not all_new_files:
        print(f"在目录 {input_dir_pattern} 中没有找到任何 CSV 文件，跳过本次合并。")
        return

    print(f"找到 {len(all_new_files)} 个新的 CSV 文件，开始读取...")

    list_new_dfs = []
    for filename in all_new_files:
        try:
            # 强制将 '股票代码' 列视为字符串
            df = pd.read_csv(filename, dtype=dtype_spec)
            
            # 严格检查列名是否匹配，如果不匹配，则尝试重命名以适应用户要求的格式
            if not all(col in df.columns for col in required_columns):
                if len(df.columns) >= len(required_columns):
                    print(f"警告: 文件 {filename} 列名不匹配，将强制使用 {required_columns}。")
                    df.columns = required_columns[:len(df.columns)]
                else:
                    print(f"错误: 文件 {filename} 列数不足，跳过。")
                    continue
            
            list_new_dfs.append(df[required_columns])
        except Exception as e:
            print(f"读取文件 {filename} 时出错: {e}")
            continue

    if not list_new_dfs:
        print("所有文件读取失败，操作终止。")
        return

    new_combined_df = pd.concat(list_new_dfs, ignore_index=True)

    # --- 2. 读取已存在的合并文件 (实现追加) ---
    existing_df = pd.DataFrame(columns=required_columns)
    if os.path.exists(output_file):
        try:
            print(f"发现已存在的合并文件 {output_file}，将进行追加。")
            # 读取旧文件时，也要保证 '股票代码' 是字符串
            existing_df = pd.read_csv(output_file, dtype=dtype_spec)
            existing_df = existing_df[required_columns]
        except Exception as e:
            print(f"读取已存在的合并文件 {output_file} 时出错: {e}. 将忽略旧数据。")

    # --- 3. 合并新数据和旧数据 ---
    final_df = pd.concat([existing_df, new_combined_df], ignore_index=True)
    initial_rows = len(final_df)

    # --- 4. 清理、去重、排序 ---
    
    # 确保 '评分' 列是数值类型
    final_df['评分'] = pd.to_numeric(final_df['评分'], errors='coerce')
    
    # 核心步骤：先按 '评分' 降序排序。这将确保在下一步去重时，评分最高的记录被保留。
    final_df.sort_values(by='评分', ascending=False, na_position='last', inplace=True)
    
    # 去重: 假设 '股票代码' 是唯一的识别字段
    final_df.drop_duplicates(subset=['股票代码'], keep='first', inplace=True)
    deduplicated_rows = len(final_df)
    print(f"原始（新+旧）记录数: {initial_rows}, 去重后记录数: {deduplicated_rows}")

    # --- 5. 保存结果 ---
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 确保保存时使用正确的列顺序，并且 **不写入索引 (index=False)**
        final_df[required_columns].to_csv(output_file, index=False, encoding='utf-8')
        print(f"成功将合并后的（去重、排序）结果保存到 {output_file}")
    except Exception as e:
        print(f"保存文件 {output_file} 时出错: {e}")

if __name__ == "__main__":
    combine_and_sort_csvs()
