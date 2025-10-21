# import akshare as ak

# stock_zygc_em_df = ak.stock_zygc_em(symbol="SH601020")
# print(stock_zygc_em_df)

# import akshare as ak

# stock_hsgt_individual_detail_em_df = ak.stock_hsgt_individual_detail_em(
# 	symbol="601020",
# 	start_date="20250225",
# 	end_date="20250425"
# )
# print(stock_hsgt_individual_detail_em_df)

import akshare as ak

stock_rank_cxg_ths_df = ak.stock_rank_cxg_ths(symbol="创月新高")
print(stock_rank_cxg_ths_df)

# 将数据保存到CSV文件
stock_rank_cxg_ths_df.to_csv("stock_new_high.csv", index=False, encoding='utf-8-sig')
print("\n创月新高数据已保存到 stock_new_high.csv")
