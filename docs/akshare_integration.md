# akshare集成方案

## 概述

为了使AGAnalysisAI系统能够分析中国A股市场，我们将集成akshare库，这是一个用于获取中国金融市场数据的Python开源库。本文档介绍了如何将akshare集成到现有系统中，并对代码进行必要的修改以支持A股分析。

## akshare简介

akshare是一个基于Python的开源金融数据接口库，专注于中国金融市场数据的获取。主要特点：

- 覆盖中国股票、债券、期货、期权、外汇等金融市场
- 提供经济数据、金融新闻、研究报告等多种数据
- 接口统一，使用简单
- 社区活跃，持续更新

官方文档：[https://akshare.akfamily.xyz/](https://akshare.akfamily.xyz/)

## 系统改造方案

### 1. 数据模型调整

需要扩展现有的数据模型，以适应A股特有的数据字段：

- 添加市场类型（沪市、深市、创业板等）
- 支持A股特有的财务指标
- 适配不同的交易规则（如涨跌停限制）

### 2. API接口适配

创建专门的A股数据获取函数，对标现有的美股API：

- `get_a_stock_prices`: 获取A股价格数据
- `get_a_stock_financial_metrics`: 获取A股财务指标
- `get_a_stock_line_items`: 获取特定财务项目
- `get_a_stock_market_cap`: 获取市值数据
- `get_a_stock_news`: 获取A股相关新闻

### 3. 分析逻辑调整

针对A股市场特点，需要调整部分分析逻辑：

- 考虑涨跌停限制对技术分析的影响
- 适配中国特色会计准则下的财务指标解读
- 增加A股特有的分析维度（如北向资金、限售股解禁等）

## 具体实现步骤

### 步骤1：安装依赖

```bash
pip install akshare pandas numpy matplotlib
```

### 步骤2：创建A股数据获取模块

创建新的模块`src/tools/akshare_api.py`，实现与现有API相似的接口，但使用akshare作为数据源。

### 步骤3：扩展现有数据模型

修改`src/data/models.py`，添加A股特有的数据字段。

### 步骤4：调整分析代理

针对中国市场特点，适当调整各个分析代理的分析逻辑。

### 步骤5：添加A股特有分析代理

考虑添加适合A股市场的特殊分析代理，如政策敏感度分析代理等。

## 数据映射关系

| 美股API数据 | akshare对应接口 | 说明 |
|-------------|----------------|------|
| 价格数据 | stock_zh_a_hist | 历史K线数据 |
| 财务指标 | stock_financial_analysis_indicator | 财务指标数据 |
| 资产负债表 | stock_financial_report_sina | 新浪财务报表 |
| 利润表 | stock_financial_report_sina | 新浪财务报表 |
| 现金流量表 | stock_financial_report_sina | 新浪财务报表 |
| 市值数据 | stock_zh_a_spot | 实时行情数据 |
| 公司新闻 | stock_news_em | 东方财富网新闻 |

## 注意事项

1. A股交易时间与美股不同，需要考虑时区转换
2. A股有涨跌停限制，影响价格波动和交易策略
3. A股财务报告披露周期与美股不同
4. 考虑中国特有的市场因素（如政策影响）
5. 数据质量和完整性可能存在差异，需要进行数据清洗和验证 