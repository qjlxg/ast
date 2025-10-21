import os
import pandas as pd
import requests
import re

from data.cache import get_cache
from data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
)

# 添加AkShare API导入
try:
    from tools.akshare_api import (
        get_a_stock_prices, 
        a_stock_prices_to_df, 
        get_a_stock_market_cap,
        get_a_stock_financial_metrics,
        search_a_stock_line_items,
        get_a_stock_news
    )
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

# Global cache instance
_cache = get_cache()


def is_chinese_stock(ticker: str) -> bool:
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


def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data from cache or API."""
    # 检查是否为中国股票
    is_a_share = is_chinese_stock(ticker)
    
    # 如果是A股且AkShare可用，使用A股价格获取函数
    if is_a_share and AKSHARE_AVAILABLE:
        return get_a_stock_prices(ticker, start_date, end_date)
    
    # 使用默认API
    # Check cache first
    if cached_data := _cache.get_prices(ticker):
        # Filter cached data by date range and convert to Price objects
        filtered_data = [Price(**price) for price in cached_data if start_date <= price["time"] <= end_date]
        if filtered_data:
            return filtered_data

    # If not in cache or no data in range, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

    # Parse response with Pydantic model
    price_response = PriceResponse(**response.json())
    prices = price_response.prices

    if not prices:
        return []

    # Cache the results as dicts
    _cache.set_prices(ticker, [p.model_dump() for p in prices])
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or API."""
    # 检查是否为中国股票
    is_a_share = is_chinese_stock(ticker)
    
    # 如果是A股且AkShare可用，使用A股API
    if is_a_share and AKSHARE_AVAILABLE:
        # 映射period参数
        a_share_period = "年报"
        if period == "quarterly":
            a_share_period = "季报"
        elif period == "ttm":
            a_share_period = "年报"  # A股使用年报替代TTM
        
        return get_a_stock_financial_metrics(ticker, end_date, a_share_period, limit)
    
    # 使用默认API
    # Check cache first
    if cached_data := _cache.get_financial_metrics(ticker):
        # Filter cached data by date and limit
        filtered_data = [FinancialMetrics(**metric) for metric in cached_data if metric["report_period"] <= end_date]
        filtered_data.sort(key=lambda x: x.report_period, reverse=True)
        if filtered_data:
            return filtered_data[:limit]

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = f"https://api.financialdatasets.ai/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

    # Parse response with Pydantic model
    metrics_response = FinancialMetricsResponse(**response.json())
    # Return the FinancialMetrics objects directly instead of converting to dict
    financial_metrics = metrics_response.financial_metrics

    if not financial_metrics:
        return []

    # Cache the results as dicts
    _cache.set_financial_metrics(ticker, [m.model_dump() for m in financial_metrics])
    return financial_metrics


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """Fetch line items from API."""
    # 检查是否为中国股票
    is_a_share = is_chinese_stock(ticker)
    
    # 如果是A股且AkShare可用，使用A股API
    if is_a_share and AKSHARE_AVAILABLE:
        # 映射period参数
        a_share_period = "年报"
        if period == "quarterly":
            a_share_period = "季报"
        elif period == "ttm":
            a_share_period = "年报"  # A股使用年报替代TTM
        
        return search_a_stock_line_items(ticker, line_items, end_date, a_share_period, limit)
    
    # 使用默认API
    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = "https://api.financialdatasets.ai/financials/search/line-items"

    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
    data = response.json()
    response_model = LineItemResponse(**data)
    search_results = response_model.search_results
    if not search_results:
        return []

    # Cache the results
    return search_results[:limit]


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or API."""
    # 检查是否为中国股票
    is_a_share = is_chinese_stock(ticker)
    
    # 如果是A股，目前暂不支持获取A股内部交易数据，返回空列表
    if is_a_share:
        print(f"暂不支持获取A股内部交易数据 ({ticker})")
        return []
    
    # 使用默认API
    # Check cache first
    if cached_data := _cache.get_insider_trades(ticker):
        # Filter cached data by date range
        filtered_data = [InsiderTrade(**trade) for trade in cached_data 
                        if (start_date is None or (trade.get("transaction_date") or trade["filing_date"]) >= start_date)
                        and (trade.get("transaction_date") or trade["filing_date"]) <= end_date]
        filtered_data.sort(key=lambda x: x.transaction_date or x.filing_date, reverse=True)
        if filtered_data:
            return filtered_data

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_trades = []
    current_end_date = end_date
    
    while True:
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
        
        data = response.json()
        response_model = InsiderTradeResponse(**data)
        insider_trades = response_model.insider_trades
        
        if not insider_trades:
            break
            
        all_trades.extend(insider_trades)
        
        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(insider_trades) < limit:
            break
            
        # Update end_date to the oldest filing date from current batch for next iteration
        current_end_date = min(trade.filing_date for trade in insider_trades).split('T')[0]
        
        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_trades:
        return []

    # Cache the results
    _cache.set_insider_trades(ticker, [trade.model_dump() for trade in all_trades])
    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """Fetch company news from cache or API."""
    # 检查是否为中国股票
    is_a_share = is_chinese_stock(ticker)
    
    # 如果是A股且AkShare可用，使用A股新闻获取函数
    if is_a_share and AKSHARE_AVAILABLE:
        return get_a_stock_news(ticker, end_date, start_date, limit)
    
    # 使用默认API
    # Check cache first
    if cached_data := _cache.get_company_news(ticker):
        # Filter cached data by date range
        filtered_data = [CompanyNews(**news) for news in cached_data 
                        if (start_date is None or news["date"] >= start_date)
                        and news["date"] <= end_date]
        filtered_data.sort(key=lambda x: x.date, reverse=True)
        if filtered_data:
            return filtered_data

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_news = []
    current_end_date = end_date
    
    while True:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
        
        data = response.json()
        response_model = CompanyNewsResponse(**data)
        company_news = response_model.news
        
        if not company_news:
            break
            
        all_news.extend(company_news)
        
        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(company_news) < limit:
            break
            
        # Update end_date to the oldest date from current batch for next iteration
        current_end_date = min(news.date for news in company_news).split('T')[0]
        
        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_news:
        return []

    # Cache the results
    _cache.set_company_news(ticker, [news.model_dump() for news in all_news])
    return all_news


def get_market_cap(
    ticker: str,
    end_date: str,
) -> float | None:
    """Fetch market cap for a given ticker."""
    # 检查是否为中国股票
    is_a_share = is_chinese_stock(ticker)
    
    # 如果是A股且AkShare可用，使用A股市值获取函数
    if is_a_share and AKSHARE_AVAILABLE:
        return get_a_stock_market_cap(ticker, end_date)
    
    # 使用默认API
    metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=1)
    if not metrics:
        return None

    return metrics[0].market_cap


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)
