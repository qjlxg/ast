import akshare as ak
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
import datetime

# 设置中文显示
# 尝试加载文泉驿字体（如果没有找到，则使用默认字体）
try:
    font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
    fontprop = FontProperties(fname=font_path)
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 定义一个全局变量，之后绘图时使用
    chinese_font = {'fontproperties': fontprop}
except:
    print("无法加载中文字体，将使用默认字体")
    chinese_font = {}

# 注意：该接口返回的数据只有最近一个交易日的有开盘价，其他日期开盘价为 0
print("正在获取A股市场数据...")
stock_zh_a_spot_em_df = ak.stock_zh_a_spot_em()
print("A股市场数据获取完成")

# 将数据存入DataFrame
df_stocks = stock_zh_a_spot_em_df

# 获取所有股票代码和名称
stock_codes = df_stocks['代码'].tolist()
stock_names = df_stocks['名称'].tolist()

print(f"\n总共获取到 {len(stock_codes)} 只股票")
print("开始分析每只股票的买入信号...")

# 获取当前日期作为查询日期
# 由于可能是在非交易日运行，我们获取昨天的数据
today = datetime.datetime.now()
yesterday = today - datetime.timedelta(days=1)
query_date = yesterday.strftime("%Y-%m-%d")

# 计算技术指标
def calculate_technical_indicators(df):
    # 计算5分钟和20分钟移动平均线
    df['MA5'] = df['收盘'].rolling(window=5).mean()
    df['MA20'] = df['收盘'].rolling(window=20).mean()
    
    # 计算相对强弱指标(RSI)
    delta = df['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 计算MACD
    df['EMA12'] = df['收盘'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['收盘'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']
    
    return df

# 交易信号分析
def analyze_signals(df):
    score = 0
    signals = []
    
    # 1. MA5与MA20交叉信号
    if df['MA5'].iloc[-1] > df['MA20'].iloc[-1] and df['MA5'].iloc[-2] <= df['MA20'].iloc[-2]:
        signals.append("MA金叉形成，可能是上涨信号")
        score += 20
    
    # 2. RSI指标分析
    current_rsi = df['RSI'].iloc[-1]
    if 30 <= current_rsi <= 50:
        signals.append("RSI处于30-50区间，有上升空间")
        score += 15
    elif current_rsi < 30:
        signals.append("RSI低于30，股票可能超卖")
        score += 20
    elif current_rsi > 70:
        signals.append("RSI高于70，股票可能超买")
        score -= 15
    
    # 3. 成交量分析
    avg_volume = df['成交量'].mean()
    current_volume = df['成交量'].iloc[-1]
    if current_volume > 1.5 * avg_volume:
        signals.append("成交量显著放大，需要关注")
        score += 10
    
    # 4. MACD分析
    if len(df) > 2 and df['Histogram'].iloc[-1] > 0 and df['Histogram'].iloc[-2] <= 0:
        signals.append("MACD金叉，买入信号")
        score += 20
    
    # 5. 价格趋势分析
    if len(df) > 5:
        price_trend = (df['收盘'].iloc[-1] - df['收盘'].iloc[-5]) / df['收盘'].iloc[-5] * 100
        if price_trend > 0 and price_trend < 3:
            signals.append("5分钟内有小幅上涨趋势")
            score += 10
        elif price_trend >= 3:
            signals.append("5分钟内有较强上涨趋势")
            score += 15
    
    return signals, score

# 创建一个列表来存储每支股票的分析结果
stock_analysis_results = []

# 遍历每个股票代码获取分钟数据并分析
count = 0
for code, name in zip(stock_codes, stock_names):
    count += 1
    print(f"\r分析进度: {count}/{len(stock_codes)}", end="")
    
    try:
        # 获取该股票的分钟级数据
        min_data = ak.stock_zh_a_hist_min_em(
            symbol=code,
            start_date=f"{query_date} 09:30:00", 
            end_date=f"{query_date} 15:00:00",
            period="1",
            adjust=""
        )
        
        if len(min_data) < 20:  # 确保有足够的数据计算技术指标
            continue
            
        # 分析数据
        df = min_data.copy()
        df = calculate_technical_indicators(df)
        signals, score = analyze_signals(df)
        
        if score > 0:  # 只记录有正面评分的股票
            stock_analysis_results.append({
                '代码': code,
                '名称': name,
                '收盘价': df['收盘'].iloc[-1],
                '信号': signals,
                '评分': score
            })
            
    except Exception as e:
        #print(f"\n获取股票 {code} 数据时出错: {str(e)}")
        continue
        
    # 加入适当的延时以避免请求过于频繁
    time.sleep(0.01)

print("\n\n分析完成，开始汇总结果...")

# 将分析结果转换为DataFrame并按评分排序
results_df = pd.DataFrame(stock_analysis_results)
if not results_df.empty:
    results_df = results_df.sort_values(by='评分', ascending=False)

    # 输出前十个值得买入的股票
    print("\n====== 前十个值得买入的股票 ======")
    top_10 = results_df.head(10)
    
    for i, (index, row) in enumerate(top_10.iterrows()):
        print(f"\n{i+1}. {row['名称']}({row['代码']}) - 评分: {row['评分']}")
        print(f"   收盘价: {row['收盘价']:.2f}")
        print(f"   买入信号:")
        for signal in row['信号']:
            print(f"   - {signal}")
    
    # 将前十名股票结果保存到CSV文件
    top_10.to_csv("top_10_stocks_to_buy.csv", index=False, encoding='utf-8-sig')
    print("\n前十名股票已保存到 top_10_stocks_to_buy.csv")
    
    # 绘制前三名股票的分析图
    for i, (index, row) in enumerate(top_10.head(3).iterrows()):
        try:
            code = row['代码']
            name = row['名称']
            
            # 重新获取数据
            min_data = ak.stock_zh_a_hist_min_em(
                symbol=code,
                start_date=f"{query_date} 09:30:00", 
                end_date=f"{query_date} 15:00:00",
                period="1",
                adjust=""
            )
            
            df = min_data.copy()
            df = calculate_technical_indicators(df)
            
            # 绘制分析图表
            plt.figure(figsize=(15, 12))
            plt.suptitle(f"{name}({code}) 技术分析图 - 评分: {row['评分']}", fontproperties=fontprop, fontsize=16)
            
            # 价格和均线
            plt.subplot(3, 1, 1)
            plt.plot(df.index, df['收盘'], label='收盘价')
            plt.plot(df.index, df['MA5'], label='5分钟MA')
            plt.plot(df.index, df['MA20'], label='20分钟MA')
            plt.title('价格与均线', **chinese_font)
            plt.legend(prop=fontprop if 'fontproperties' in chinese_font else None)
            
            # RSI指标
            plt.subplot(3, 1, 2)
            plt.plot(df.index, df['RSI'], label='RSI')
            plt.axhline(y=70, color='r', linestyle='--')
            plt.axhline(y=30, color='g', linestyle='--')
            plt.title('RSI指标', **chinese_font)
            plt.legend(prop=fontprop if 'fontproperties' in chinese_font else None)
            
            # MACD指标
            plt.subplot(3, 1, 3)
            plt.plot(df.index, df['MACD'], label='MACD')
            plt.plot(df.index, df['Signal'], label='Signal')
            plt.bar(df.index, df['Histogram'], label='Histogram')
            plt.title('MACD指标', **chinese_font)
            plt.legend(prop=fontprop if 'fontproperties' in chinese_font else None)
            
            plt.tight_layout()
            plt.savefig(f'{code}_{name}_analysis.png')
            print(f"\n{name}({code})的分析图表已保存")
            
        except Exception as e:
            print(f"\n绘制{name}({code})图表时出错: {str(e)}")
            
else:
    print("\n没有找到符合买入条件的股票")



