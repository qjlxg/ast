# 获取股票的买卖盘口数据
# import akshare as ak

# stock_bid_ask_em_df = ak.stock_bid_ask_em(symbol="601020")
# print(stock_bid_ask_em_df)

# 获取股票的K线数据
import akshare as ak
stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="601020", start_date="20250101", end_date="20250423")
print(stock_zh_a_hist_df)

# 导入matplotlib库用于绘图
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# 设置中文显示
# 尝试使用多种可能的中文字体
try:
    # 优先使用文泉驿正黑
    font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
    font_prop = FontProperties(fname=font_path)
except:
    # 如果上面的不可用，使用matplotlib的内置方法
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Zen Hei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    font_prop = None

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
plt.figure(figsize=(12, 6))

# 绘制收盘价曲线
plt.plot(stock_zh_a_hist_df['日期'], stock_zh_a_hist_df['收盘'], label='收盘价')

# 设置图表标题和轴标签
if font_prop:
    plt.title('601020股票价格走势图', fontproperties=font_prop)
    plt.xlabel('日期', fontproperties=font_prop)
    plt.ylabel('价格(元)', fontproperties=font_prop)
    plt.legend(prop=font_prop)
else:
    plt.title('601020股票价格走势图')
    plt.xlabel('日期')
    plt.ylabel('价格(元)')
    plt.legend()

# 旋转x轴日期标签，避免重叠
plt.xticks(rotation=45)

# 优化布局
plt.tight_layout()

# 保存图像文件（替代显示）
plt.savefig('stock_price_chart.png')
print("图表已保存为 stock_price_chart.png")

# 显示图形
plt.show()



