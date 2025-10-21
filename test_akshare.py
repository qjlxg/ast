"""
测试akshare功能是否正常工作的脚本
用法：python test_akshare.py 600000
"""

import sys
import akshare as ak
import pandas as pd
import datetime

def test_basic_stock_info(code):
    """测试基本股票信息获取功能"""
    print(f"\n=== 测试基本股票信息 ({code}) ===")
    
    # 尝试获取个股信息
    try:
        print("尝试 stock_individual_info_em...")
        stock_info = ak.stock_individual_info_em(symbol=code)
        print(f"获取成功，共 {len(stock_info)} 行数据")
        print(stock_info.head())
    except Exception as e:
        print(f"错误: {e}")
        
    # 尝试获取实时行情
    try:
        print("\n尝试 stock_zh_a_spot...")
        spot_data = ak.stock_zh_a_spot()
        print(f"获取成功，共 {len(spot_data)} 行数据")
        # 查找特定股票
        stock_row = spot_data[spot_data['代码'] == code]
        if not stock_row.empty:
            print(f"找到股票 {code}:")
            print(stock_row)
        else:
            print(f"未找到股票 {code}")
    except Exception as e:
        print(f"错误: {e}")


def test_historical_data(code):
    """测试历史数据获取功能"""
    print(f"\n=== 测试历史数据 ({code}) ===")
    
    # 准备日期参数
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    
    # 尝试获取历史行情
    try:
        print(f"尝试 stock_zh_a_hist ({start_date} 至 {end_date})...")
        
        # 对于纯数字代码，尝试添加市场前缀
        if code.isdigit():
            if code.startswith('6'):
                market_code = f"sh{code}"
            else:
                market_code = f"sz{code}"
        else:
            market_code = code
            
        hist_data = ak.stock_zh_a_hist(symbol=market_code, start_date=start_date, end_date=end_date)
        print(f"获取成功，共 {len(hist_data)} 行数据")
        if not hist_data.empty:
            print(hist_data.head())
    except Exception as e:
        print(f"错误: {e}")


def test_financial_data(code):
    """测试财务数据获取功能"""
    print(f"\n=== 测试财务数据 ({code}) ===")
    
    # 尝试获取财务指标
    try:
        print("尝试 stock_financial_analysis_indicator...")
        financial_data = ak.stock_financial_analysis_indicator(stock=code)
        print(f"获取成功，共 {len(financial_data)} 行数据")
        if not financial_data.empty:
            print(financial_data.head())
    except Exception as e:
        print(f"错误: {e}")
        
    # 尝试获取财务报表
    try:
        print("\n尝试 stock_financial_report_sina (资产负债表)...")
        balance_sheet = ak.stock_financial_report_sina(stock=code, symbol="资产负债表")
        print(f"获取成功，共 {len(balance_sheet)} 行数据")
        if not balance_sheet.empty:
            print(balance_sheet.head())
    except Exception as e:
        print(f"错误: {e}")


def test_market_data(code):
    """测试市场数据获取功能"""
    print(f"\n=== 测试市场数据 ===")
    
    # 尝试获取北向资金数据
    try:
        print("尝试 stock_hk_ggt_components_em...")
        north_data = ak.stock_hk_ggt_components_em()
        print(f"获取成功，共 {len(north_data)} 行数据")
        if not north_data.empty:
            stock_data = north_data[north_data['代码'] == code]
            if not stock_data.empty:
                print(f"找到股票 {code}:")
                print(stock_data)
            else:
                print(f"未找到股票 {code} 的北向资金数据")
    except Exception as e:
        print(f"错误: {e}")
        
    # 尝试获取行业板块数据
    try:
        print("\n尝试 stock_sector_spot...")
        sector_data = ak.stock_sector_spot()
        print(f"获取成功，共 {len(sector_data)} 行数据")
        if not sector_data.empty:
            print(sector_data.head())
    except Exception as e:
        print(f"错误: {e}")
        
    # 尝试获取融资融券数据
    try:
        print("\n尝试可用的融资融券数据接口...")
        available_funcs = []
        
        # 尝试多个可能的函数
        try:
            data = ak.stock_margin_detail_em(symbol=code)
            if not data.empty:
                available_funcs.append("stock_margin_detail_em")
        except:
            pass
            
        try:
            if code.startswith('6'):
                data = ak.stock_margin_sh_detail_em(symbol=code)
                if not data.empty:
                    available_funcs.append("stock_margin_sh_detail_em")
            else:
                data = ak.stock_margin_sz_detail_em(symbol=code)
                if not data.empty:
                    available_funcs.append("stock_margin_sz_detail_em")
        except:
            pass
            
        try:
            current_month = datetime.datetime.now().strftime("%Y%m")
            data = ak.stock_margin_underlying_info_szse(date=current_month)
            if not data.empty:
                available_funcs.append("stock_margin_underlying_info_szse")
        except:
            pass
            
        print(f"可用的融资融券函数: {', '.join(available_funcs) if available_funcs else '无'}")
        
    except Exception as e:
        print(f"错误: {e}")
        

def test_all(code):
    """运行所有测试"""
    print(f"开始测试akshare功能，股票代码: {code}")
    
    # 移除可能的市场前缀
    if code.startswith(('sh', 'sz')):
        pure_code = code[2:]
    else:
        pure_code = code
    
    test_basic_stock_info(pure_code)
    test_historical_data(pure_code)
    test_financial_data(pure_code)
    test_market_data(pure_code)
    
    print("\n测试完成!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python test_akshare.py 股票代码")
        print("例如: python test_akshare.py 600000")
        sys.exit(1)
        
    stock_code = sys.argv[1]
    test_all(stock_code) 