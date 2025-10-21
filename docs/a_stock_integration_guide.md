# A股分析系统集成指南

## 简介

本指南详细介绍了如何将AGAnalysisAI系统扩展为支持中国A股市场分析。通过集成akshare库和添加中国特色分析代理，我们可以对A股进行全面的AI驱动分析。

## 准备工作

### 依赖安装

```bash
# 安装akshare及其依赖
pip install akshare pandas numpy matplotlib tushare

# 更新requirements.txt
echo "akshare>=1.0.0" >> requirements.txt
echo "tushare>=1.2.0" >> requirements.txt
```

### 配置修改

将以下内容添加到`.env`文件中：

```
# A股API配置
TUSHARE_API_KEY=your_tushare_api_key  # 可选，用于增加数据源
```

## 集成步骤

### 1. 添加A股数据获取模块

1. 创建`src/tools/akshare_api.py`文件，实现A股数据获取功能
2. 该模块提供了与现有美股API接口兼容的函数，但使用akshare作为数据源

### 2. 添加中国特色分析代理

1. 创建`src/agents/china_market.py`文件，实现专门分析中国市场特色因素的代理
2. 该代理分析北向资金、限售股解禁、行业板块、融资融券和政策敏感度等A股特有因素

### 3. 修改工作流配置

在`src/utils/analysts.py`中注册新的中国市场分析师：

```python
# 添加到ANALYST_ORDER列表
ANALYST_ORDER = [
    # 现有分析师...
    ("中国市场分析 (China Market)", "china_market_agent"),
]

# 更新get_analyst_nodes函数
def get_analyst_nodes():
    return {
        # 现有分析师...
        "china_market_agent": ("china_market_agent", china_market_agent),
    }
```

### 4. 导入新模块

在`src/main.py`中添加新的导入：

```python
from agents.china_market import china_market_agent
from tools.akshare_api import (
    get_a_stock_prices,
    get_a_stock_financial_metrics,
    search_a_stock_line_items,
    get_a_stock_market_cap,
    get_a_stock_news
)
```

## 使用方法

### A股代码输入格式

A股代码可以采用以下格式：

1. 纯数字代码：`600000`（系统会自动识别沪深市场）
2. 带市场前缀：`sh600000`或`sz000001`
3. 带点的代码：`600000.SH`或`000001.SZ`（系统会自动转换）

### 运行系统分析A股

```bash
# 分析单只A股
python src/main.py --tickers 600000 --end-date 2023-12-31

# 分析多只A股
python src/main.py --tickers 600000,000001,300059 --start-date 2023-01-01 --end-date 2023-12-31 --show-reasoning
```

## 扩展功能

### 添加A股特有技术指标

在`src/agents/technicals.py`中可以添加A股特有的技术指标，如：

1. 北向资金指标
2. 涨跌停板分析
3. 量能指标
4. 板块轮动指标

### 添加宏观经济分析

可以创建专门的宏观经济分析代理，关注：

1. 中国GDP增长率
2. CPI/PPI走势
3. 货币政策变化
4. 财政政策调整
5. 国际贸易数据

## 注意事项

1. **时区处理**：A股交易时间为北京时间9:30-15:00，与美股交易时间不同
2. **交易规则**：A股有涨跌停限制，需要在策略中考虑
3. **数据频率**：A股财务数据通常为季度发布，与美股可能有差异
4. **中文处理**：A股相关数据大多为中文，确保系统能正确处理中文字符
5. **假期安排**：A股有特殊的节假日休市安排，需要在时间序列分析中考虑

## 故障排除

### 数据获取问题

如果遇到数据获取问题，请检查：

1. 网络连接是否正常
2. akshare版本是否最新
3. 股票代码格式是否正确
4. API调用频率是否过高导致限流

### 分析结果异常

如果分析结果异常，请检查：

1. 输入的日期范围是否有效（考虑节假日）
2. 股票是否处于特殊状态（如停牌、退市等）
3. 是否使用了不适用于A股的分析方法

## 贡献和改进

随着项目的发展，我们计划进一步增强A股分析能力：

1. 添加更多的A股特有指标
2. 改进北向资金分析
3. 增加基于中国特色会计准则的财务分析
4. 开发专门的政策风险评估模型
5. 集成更多中国特色的分析师策略

欢迎提交PR或问题反馈，共同改进A股分析系统！ 