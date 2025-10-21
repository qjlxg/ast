import akshare as ak
import time

# 注意：该接口返回的数据只有最近一个交易日的有开盘价，其他日期开盘价为 0
stock_zh_a_spot_em_df = ak.stock_zh_a_spot_em()
print(stock_zh_a_spot_em_df)

# 将数据存入DataFrame
df_stocks = stock_zh_a_spot_em_df

# 获取所有股票代码
stock_codes = df_stocks['代码'].tolist()

print(f"\n总共获取到 {len(stock_codes)} 只股票")
print("开始获取每只股票的分钟级数据...")

# 遍历每个股票代码获取分钟数据
for code in stock_codes:  # 这里只取前3个用于测试,实际使用时去掉[:3]
    try:
        # 获取该股票的分钟级数据
        print(f"\n正在获取股票 {code} 的分钟数据...")
        min_data = ak.stock_zh_a_hist_min_em(
            symbol=code,
            start_date="2025-04-23 09:30:00", 
            end_date="2025-04-23 14:00:00",
            period="1",
            adjust=""
        )
        print(f"成功获取到 {len(min_data)} 条记录")
        
    except Exception as e:
        print(f"获取股票 {code} 数据时出错: {str(e)}")
        continue
        
    # 加入适当的延时以避免请求过于频繁
    time.sleep(1)

stock_zh_a_hist_min_em_df = ak.stock_zh_a_hist_min_em(symbol="601020", start_date="2025-04-23 09:30:00", end_date="2025-04-23 14:00:00", period="1", adjust="")
print(stock_zh_a_hist_min_em_df)

# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties

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
    
    return df

# 分析数据
df = stock_zh_a_hist_min_em_df.copy()
df = calculate_technical_indicators(df)

# 交易信号分析
def analyze_signals(df):
    signals = []
    
    # 1. MA5与MA20交叉信号
    if df['MA5'].iloc[-1] > df['MA20'].iloc[-1] and df['MA5'].iloc[-2] <= df['MA20'].iloc[-2]:
        signals.append("MA金叉形成，可能是上涨信号")
    
    # 2. RSI指标分析
    current_rsi = df['RSI'].iloc[-1]
    if current_rsi < 30:
        signals.append("RSI低于30，股票可能超卖")
    elif current_rsi > 70:
        signals.append("RSI高于70，股票可能超买")
        
    # 3. 成交量分析
    avg_volume = df['成交量'].mean()
    current_volume = df['成交量'].iloc[-1]
    if current_volume > 1.5 * avg_volume:
        signals.append("成交量显著放大，需要关注")
    
    return signals

# 执行分析并打印结果
signals = analyze_signals(df)
print("\n=== 交易策略分析结果 ===")
print(f"最新收盘价: {df['收盘'].iloc[-1]:.2f}")
print(f"5分钟均线: {df['MA5'].iloc[-1]:.2f}")
print(f"20分钟均线: {df['MA20'].iloc[-1]:.2f}")
print(f"RSI指标: {df['RSI'].iloc[-1]:.2f}")
print("\n交易信号:")
for signal in signals:
    print(f"- {signal}")

# 绘制分析图表
plt.figure(figsize=(15, 10))

# 价格和均线
plt.subplot(2, 1, 1)
plt.plot(df.index, df['收盘'], label='收盘价')
plt.plot(df.index, df['MA5'], label='5分钟MA')
plt.plot(df.index, df['MA20'], label='20分钟MA')
plt.title('股票分时走势与技术指标', **chinese_font)
plt.legend(prop=fontprop if 'fontproperties' in chinese_font else None)

# RSI指标
plt.subplot(2, 1, 2)
plt.plot(df.index, df['RSI'], label='RSI')
plt.axhline(y=70, color='r', linestyle='--')
plt.axhline(y=30, color='g', linestyle='--')
plt.legend(prop=fontprop if 'fontproperties' in chinese_font else None)

plt.tight_layout()
plt.savefig('technical_analysis.png')
print("\n分析图表已保存为 technical_analysis.png")

# 给出建议
print("\n=== 交易建议 ===")
if len(signals) > 0:
    print("根据技术分析结果，建议：")
    if "MA金叉形成" in str(signals) or "RSI低于30" in str(signals):
        print("可以考虑买入，但需要结合其他基本面因素")
    elif "RSI高于70" in str(signals):
        print("当前价格可能偏高，建议观望")
    else:
        print("建议继续观察市场动向")
else:
    print("当前无明显交易信号，建议观望")


