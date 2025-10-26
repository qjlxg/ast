import os
import pandas as pd
from datetime import datetime
import pytz

# --- 配置 ---
INPUT_DIR = "stock_data"
OUTPUT_DIR = "results"
TIMEZONE = "Asia/Shanghai"  # 上海时区

# --- 技术指标计算函数 ---
def calculate_macd(df, short_period=12, long_period=26, signal_period=9):
    """计算 MACD 指标 (DIF, DEA, MACD 柱)"""
    # 确保 '收盘' 列是数字类型
    df['收盘'] = pd.to_numeric(df['收盘'], errors='coerce')

    # 计算 EMA
    df['EMA_short'] = df['收盘'].ewm(span=short_period, adjust=False).mean()
    df['EMA_long'] = df['收盘'].ewm(span=long_period, adjust=False).mean()

    # 计算 DIF (快线)
    df['DIF'] = df['EMA_short'] - df['EMA_long']

    # 计算 DEA (慢线 / 信号线)
    df['DEA'] = df['DIF'].ewm(span=signal_period, adjust=False).mean()

    # 计算 MACD 柱 (柱状图)
    df['MACD'] = (df['DIF'] - df['DEA']) * 2
    return df

def calculate_kdj(df, n=9, m1=3, m2=3):
    """计算 KDJ 指标 (RSV, K, D, J)"""
    # 确保 '最高', '最低', '收盘' 列是数字类型
    df['最高'] = pd.to_numeric(df['最高'], errors='coerce')
    df['最低'] = pd.to_numeric(df['最低'], errors='coerce')
    df['收盘'] = pd.to_numeric(df['收盘'], errors='coerce')

    # 1. 计算 RSV
    low_list = df['最低'].rolling(window=n).min()
    high_list = df['最高'].rolling(window=n).max()
    df['RSV'] = (df['收盘'] - low_list) / (high_list - low_list) * 100

    # 2. 计算 K (简单移动平均)
    df['K'] = df['RSV'].ewm(com=m1 - 1, adjust=False).mean()

    # 3. 计算 D (K 的移动平均)
    df['D'] = df['K'].ewm(com=m2 - 1, adjust=False).mean()

    # 4. 计算 J
    df['J'] = 3 * df['K'] - 2 * df['D']
    return df

# --- 核心筛选逻辑 ---
def check_right_side_pattern(df):
    """
    检查右侧模式条件：
    1. MACD 抬起多头形态下的上升形态
    2. KDJ 的 J 值做死叉形态
    3. 保证 MACD 趋势没有走坏 (DIF 和 DEA 保持在 0 轴之上)
    
    返回满足条件的最新交易日数据
    """
    # 确保有足够的历史数据进行计算
    if len(df) < 30:
        return None

    # 获取最后一天的数据 (即最新交易日)
    latest = df.iloc[-1]
    
    # 获取前一天的数据
    prev = df.iloc[-2]

    # --- 条件 1 & 3: MACD 趋势判断 (多头 & 未走坏) ---
    macd_bullish_trend = (latest['DIF'] > latest['DEA']) and \
                         (latest['DIF'] > 0 and latest['DEA'] > 0)
    
    # MACD 趋势向上 (多头形态下的上升形态)
    macd_rising = (latest['DIF'] > prev['DIF']) and (latest['DEA'] > prev['DEA'])

    # --- 条件 2: KDJ J 值死叉形态 ---
    # J 值死叉形态：J 值从前一天的较高水平回落，且 J 值小于等于 K 值 (J > K 变为 J <= K，且 J 值小于 90 以避免在极高位回调)
    j_dead_cross = (prev['J'] > prev['K']) and (latest['J'] <= latest['K']) and (latest['J'] < 90)
    
    # 结合判断
    if macd_bullish_trend and macd_rising and j_dead_cross:
        # 满足条件的点视为“右侧大方向回调到位形态没有破坏掉的加仓点”
        return {
            '日期': latest['日期'],
            '收盘': latest['收盘'],
            'MACD_DIF': latest['DIF'],
            'MACD_DEA': latest['DEA'],
            'KDJ_J': latest['J'],
            'KDJ_K': latest['K'],
            'KDJ_D': latest['D'],
            '说明': 'MACD多头趋势上升，KDJ J值死叉回调，满足右侧加仓条件'
        }
    return None

# --- 主执行逻辑 ---
def main():
    stock_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    all_results = []
    
    # 检查数据目录是否存在
    if not os.path.exists(INPUT_DIR) or not stock_files:
        print(f"错误: 找不到 {INPUT_DIR} 目录或其中没有 CSV 文件。请确认您的数据已正确上传。")
        # 在 GitHub Actions 中，如果找不到数据，允许脚本继续运行但不会产生结果
        return

    for filename in stock_files:
        filepath = os.path.join(INPUT_DIR, filename)
        stock_code = filename.replace('.csv', '')
        
        try:
            # 读取数据
            df = pd.read_csv(filepath)
            
            # 确保数据不为空
            if df.empty:
                print(f"警告: 文件 {filename} 为空，跳过处理。")
                continue

            # 将 '日期' 转换为 datetime 对象并排序
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values(by='日期').reset_index(drop=True)
            
            # 计算指标
            df = calculate_macd(df)
            df = calculate_kdj(df)
            
            # 检查模式
            result = check_right_side_pattern(df)
            
            if result:
                result['股票代码'] = stock_code
                all_results.append(result)
                
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

    # --- 结果输出 ---
    if not all_results:
        print("未发现符合右侧模式的股票。")
        return

    # 转换为 DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 获取当前时间（上海时区）
    shanghai_tz = pytz.timezone(TIMEZONE)
    now_shanghai = datetime.now(shanghai_tz)
    
    # 格式化时间戳和路径
    timestamp_str = now_shanghai.strftime("%Y%m%d%H%M%S")
    year_month_dir = now_shanghai.strftime("%Y%m")
    
    # 创建输出目录
    final_output_dir = os.path.join(OUTPUT_DIR, year_month_dir)
    os.makedirs(final_output_dir, exist_ok=True)
    
    # 构造输出文件名
    output_filename = f"right_side_pattern_{timestamp_str}.csv"
    output_path = os.path.join(final_output_dir, output_filename)
    
    # 保存结果
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"成功筛选出 {len(all_results)} 条记录，已保存至: {output_path}")

if __name__ == "__main__":
    # 创建必要目录（在本地或Actions中）
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 仅在非 Actions 环境下提醒用户
    if 'GITHUB_ACTIONS' not in os.environ:
        print(f"股票数据期望在 `{INPUT_DIR}/` 目录中。")
    main()
