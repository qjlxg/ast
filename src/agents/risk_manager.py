from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
from tools.api import get_prices, prices_to_df
import json
import re
import pandas as pd
from datetime import datetime

# 添加A股API导入
try:
    from tools.akshare_api import get_a_stock_prices, a_stock_prices_to_df, get_a_stock_market_cap
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

##### Risk Management Agent #####
def risk_management_agent(state: AgentState):
    """Controls position sizing based on real-world risk factors for multiple tickers."""
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    tickers = data["tickers"]

    # Initialize risk analysis for each ticker
    risk_analysis = {}
    current_prices = {}  # Store prices here to avoid redundant API calls

    for ticker in tickers:
        progress.update_status("risk_management_agent", ticker, "Analyzing price data")

        # 检查是否为A股代码
        is_a_stock = is_chinese_stock(ticker)
        
        if is_a_stock and AKSHARE_AVAILABLE:
            # 使用A股数据API
            prices = get_a_stock_prices(
                ticker=ticker,
                start_date=data["start_date"],
                end_date=data["end_date"],
            )
            if prices:
                prices_df = a_stock_prices_to_df(prices)
            else:
                prices_df = None
        else:
            # 使用美股数据API
            prices = get_prices(
                ticker=ticker,
                start_date=data["start_date"],
                end_date=data["end_date"],
            )
            if prices:
                prices_df = prices_to_df(prices)
            else:
                prices_df = None

        # 修复：检查DataFrame是否为None或为空
        if prices_df is None or (isinstance(prices_df, pd.DataFrame) and prices_df.empty):
            progress.update_status("risk_management_agent", ticker, "Failed: No price data found")
            risk_analysis[ticker] = {
                "remaining_position_limit": 0.0,
                "current_price": 0.0,
                "reasoning": {
                    "error": "No price data available for this ticker"
                },
            }
            continue

        progress.update_status("risk_management_agent", ticker, "Calculating position limits")

        # Calculate portfolio value
        current_price = prices_df["close"].iloc[-1]
        current_prices[ticker] = current_price  # Store the current price

        # Calculate current position value for this ticker
        current_position_value = portfolio.get("cost_basis", {}).get(ticker, 0)

        # Calculate total portfolio value using stored prices
        total_portfolio_value = portfolio.get("cash", 0) + sum(portfolio.get("cost_basis", {}).get(t, 0) for t in portfolio.get("cost_basis", {}))

        # Base limit is 20% of portfolio for any single position
        position_limit = total_portfolio_value * 0.20

        # For existing positions, subtract current position value from limit
        remaining_position_limit = position_limit - current_position_value

        # Ensure we don't exceed available cash
        max_position_size = min(remaining_position_limit, portfolio.get("cash", 0))

        risk_analysis[ticker] = {
            "remaining_position_limit": float(max_position_size),
            "current_price": float(current_price),
            "reasoning": {
                "portfolio_value": float(total_portfolio_value),
                "current_position": float(current_position_value),
                "position_limit": float(position_limit),
                "remaining_limit": float(remaining_position_limit),
                "available_cash": float(portfolio.get("cash", 0)),
            },
        }

        progress.update_status("risk_management_agent", ticker, "Done")

    message = HumanMessage(
        content=json.dumps(risk_analysis),
        name="risk_management_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(risk_analysis, "Risk Management Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["risk_management_agent"] = risk_analysis

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }

def is_chinese_stock(ticker):
    """
    判断是否为中国股票代码
    支持以下格式：
    1. 以sh或sz开头: sh600000, sz000001
    2. 6位数字开头为沪市，0或3开头为深市: 600000, 000001, 300059
    3. 带后缀的代码: 600000.SH, 000001.SZ
    """
    # 检查是否以sh或sz开头
    if ticker.startswith(('sh', 'sz', 'bj')):
        return True
    
    # 检查是否为纯数字代码
    if re.match(r'^[0-9]{6}$', ticker):
        return True
    
    # 检查是否为带后缀的代码
    if re.match(r'^[0-9]{6}\.(SH|SZ|BJ)$', ticker, re.IGNORECASE):
        return True
    
    return False

def analyze_risk(self, ticker: str, price_data: pd.DataFrame, risk_free_rate: float = 0.05):
    """
    分析给定股票的风险情况
    参数:
        ticker: 股票代码
        price_data: 价格数据
        risk_free_rate: 无风险利率
    返回:
        风险分析结果
    """
    try:
        # 根据代码判断是美股还是A股
        is_a_share = self.is_chinese_stock(ticker)
        
        # 计算基本指标
        daily_returns = price_data['close'].pct_change().dropna()
        if len(daily_returns) == 0:
            return {
                "message": "无法分析风险 - 价格数据不足",
                "risk_level": "未知",
                "details": {}
            }
                
        volatility = daily_returns.std() * (252 ** 0.5)  # 年化波动率
        max_drawdown = self.calculate_max_drawdown(price_data['close'])
        beta = self.calculate_beta(ticker, price_data, is_a_share)
        
        # 获取市值
        market_cap = None
        if is_a_share:
            # 使用A股市值数据
            market_cap = self.get_a_share_market_cap(ticker)
        else:
            # 使用美股市值数据
            market_cap = self.get_market_cap(ticker)
        
        # 分析风险水平
        risk_score = 0
        risk_factors = []
        
        # 波动率评估（波动率越高，风险越大）
        if volatility > 0.4:
            risk_score += 3
            risk_factors.append(f"波动率非常高 ({volatility:.2f})")
        elif volatility > 0.25:
            risk_score += 2
            risk_factors.append(f"波动率高 ({volatility:.2f})")
        elif volatility > 0.15:
            risk_score += 1
            risk_factors.append(f"波动率中等 ({volatility:.2f})")
        
        # 最大回撤评估
        if max_drawdown > 0.4:
            risk_score += 3
            risk_factors.append(f"最大回撤非常高 ({max_drawdown:.2f})")
        elif max_drawdown > 0.25:
            risk_score += 2
            risk_factors.append(f"最大回撤高 ({max_drawdown:.2f})")
        elif max_drawdown > 0.15:
            risk_score += 1
            risk_factors.append(f"最大回撤中等 ({max_drawdown:.2f})")
        
        # Beta评估（Beta越高，风险越大）
        if beta is not None:
            if beta > 1.5:
                risk_score += 2
                risk_factors.append(f"Beta系数高 ({beta:.2f})")
            elif beta > 1.0:
                risk_score += 1
                risk_factors.append(f"Beta系数略高 ({beta:.2f})")
            elif beta < 0:
                risk_score += 1
                risk_factors.append(f"Beta系数为负 ({beta:.2f})，与市场逆相关")
        
        # 市值评估（市值越小，风险越大）
        if market_cap is not None:
            market_cap_billions = market_cap / 1_000_000_000
            if market_cap_billions < 1:
                risk_score += 3
                risk_factors.append(f"市值小 ({market_cap_billions:.2f}B)，流动性风险高")
            elif market_cap_billions < 5:
                risk_score += 2
                risk_factors.append(f"市值中等 ({market_cap_billions:.2f}B)，存在一定流动性风险")
            elif market_cap_billions < 20:
                risk_score += 1
                risk_factors.append(f"市值较大 ({market_cap_billions:.2f}B)")
        else:
            risk_factors.append("无法获取市值数据")
        
        # 风险水平评定
        risk_level = "低"
        if risk_score >= 7:
            risk_level = "极高"
        elif risk_score >= 5:
            risk_level = "高"
        elif risk_score >= 3:
            risk_level = "中等"
        elif risk_score >= 1:
            risk_level = "较低"
        
        # 计算夏普比率
        avg_return = daily_returns.mean() * 252
        sharpe_ratio = (avg_return - risk_free_rate) / volatility if volatility != 0 else 0
        
        # 输出分析结果
        result = {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "details": {
                "volatility": volatility,
                "max_drawdown": max_drawdown,
                "beta": beta,
                "market_cap": market_cap,
                "sharpe_ratio": sharpe_ratio
            }
        }
        
        return result
    except Exception as e:
        return {
            "message": f"风险分析失败: {str(e)}",
            "risk_level": "未知",
            "details": {}
        }

def get_a_share_market_cap(self, ticker: str) -> float:
    """获取A股市值数据"""
    try:
        # 获取最新日期作为查询日期
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        # 使用akshare API获取市值
        market_cap = get_a_stock_market_cap(ticker, end_date)
        
        return market_cap
    except Exception as e:
        print(f"获取A股市值出错 ({ticker}): {e}")
        return None
