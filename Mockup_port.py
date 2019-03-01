#%% Import Libs
import pandas as pd
pd.options.mode.chained_assignment = None  
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
# from pony.orm import *
import xlwings as xw
import datetime

# def main(): 
# 	wb = xw.books('Mockup_port.xlsb')
# 	port_list = wb.sheets('Portfolios').range('port_list').value
# 	wb.sheets('Portfolios').range('P1').value = port_list

wb = xw.books('Mockup_port.xlsb')
port_list = tuple(wb.sheets('Portfolios').range('port_list').value)
# wb.sheets('Portfolios').range('P1').value = np.asarray(port_list).reshape(-1,1)

bg_date = wb.sheets('Main').range("B1").value 
ed_date = wb.sheets('Main').range("D1").value 

conn = psycopg2.connect(user="postgres", 
		password="root", 
		host="10.56.35.166",
		port="5433",
		database="PortAnalytic-2016-17v14")

# cursor = conn.cursor()
cursor_bos = conn.cursor(cursor_factory=RealDictCursor)
cursor_pa = conn.cursor(cursor_factory=RealDictCursor)

#Query from BOS 
# cursor_bos.execute('''
# 	SELECT bs.*, pa."HoldDate", pa."ParentPortfolioCode", pa."PortfolioCode", pa."HLD_weight", pa."HoldingType" FROM dbo."PA_Holding" pa 
# 	INNER JOIN dbo."BOS_Holding" bs ON pa."Id" = bs."Id"
# 	WHERE pa."ParentPortfolioCode" IN %s
# 	AND pa."HoldDate" BETWEEN %s AND %s ORDER BY pa."ParentPortfolioCode" ASC;
# ''', (port_list, bg_date, ed_date))
# cursor_bos.execute('''
# 	SELECT Foreign_trans.*, fx."CurrencyRate" from dbo."PA_FXrate" fx
# 	RIGHT OUTER JOIN 
# 		(SELECT bs.*, pa.*,bss.* FROM dbo."PA_Holding" pa 
# 		INNER JOIN dbo."BOS_Holding" bs ON pa."Id" = bs."Id"
# 		INNER JOIN dbo."BOS_Security" bss ON pa."Id" = bss."Id"
# 		WHERE pa."ParentPortfolioCode" IN (SELECT bp."LinkPortfolioCode" FROM dbo."BOS_Portfolio" bp WHERE bp."Status" = 'A' AND bp."IsTermedFund" = FALSE)
# 		AND pa."HoldDate" BETWEEN %s AND %s
# 		AND pa."ParentPortfolioCode" IN %s
# 		AND bs."Group1" NOT IN ('NET ASSET VALUE PER UNIT', 'INVESTMENT UNIT', 'NET ASSET VALUE')) AS Foreign_trans 
# 	ON fx."HoldDate" = Foreign_trans."HoldDate" AND fx."CurrencyCode" = Foreign_trans."Currency";

# ''',(bg_date, ed_date, port_list))

cursor_bos.execute('''
	SELECT pa.*, pd.*, bos.* FROM dbo."BOS_Holding" bos
	RIGHT JOIN dbo."PA_Holding" pa ON pa."Id" = bos."Id"
	LEFT JOIN dbo."BOS_Derivative" pd ON pd."Id" = bos."Id"
	WHERE pa."ParentPortfolioCode" IN %s
	AND pa."HoldDate" BETWEEN %s AND %s ORDER BY pa."ParentPortfolioCode" ASC;
''',(port_list, bg_date, ed_date))

#Query from Portia
cursor_pa.execute('''
    SELECT pt.*, pa.* FROM dbo."PORTIA_Transaction" pt
    INNER JOIN dbo."PA_Portfolio" pa ON pt."ParentPortfolioCode" = pa."Code"
    WHERE pt."SettleDate" = %s;
''', (ed_date,))

bos_record = cursor_bos.fetchall()
pa_record = cursor_pa.fetchall()

temp_df = pd.DataFrame(bos_record)
# col_order = ['HoldDate', 'ParentPortfolioCode', 'PortfolioCode', 'Group1', 'Group2', 'Group3',
# 	'HLD_SecRating', 'HLD_Issuer', 'HLD_IssType', 'HLD_IssIndustry', 'HLD_Rating', 'HLD_Aval',
# 	'HLD_Recourse', 'HLD_MaturityDate', 'HLD_SecDTM', 'HLD_CouponRate', 'HLD_Yield', 'HLD_Quantity',
# 	'HLD_Price', 'HLD_MarketValue', 'HLD_AI', 'HLD_Cost', 'HLD_ModDuration', 'HLD_Principal',
# 	'HLD_Remark', 'HLD_PriceCost', 'HLD_YieldCost', 'HLD_ratio', 'HLD_weight'
# 	]
# temp_df = temp_df[col_order]

#%% to workbook
wb.sheets('Raw_BOS').range('A5').value = temp_df.values
wb.sheets('Raw_BOS').range('A4').value = temp_df.columns.values

#%% Cal one port 
cal_port = 'SCBFPFUND'
bg_transaction = temp_df[(temp_df['ParentPortfolioCode'] == cal_port) & (temp_df['HoldDate']==bg_date)]
ed_transaction = temp_df[(temp_df['ParentPortfolioCode'] == cal_port) & (temp_df['HoldDate']==ed_date)]


#%% Flag by sec 
group_col = ['ParentPortfolioCode', 'PortfolioCode', 'Group1', 'Group2', 'Group3', 'HLD_SecRating', 'HLD_Issuer',
	'HLD_IssType', 'HLD_IssIndustry', 'HLD_Rating', 'HLD_Aval', 'HLD_Recourse', 'HLD_MaturityDate']

exclude_list = ['NET ASSET VALUE PER UNIT', 'INVESTMENT UNIT', 'NET ASSET VALUE', 'MANAGEMENT FEE',
	'CUSTODIAN FEE', 'NET ASSET BEFORE FEES', 'MANAGEMENT FEE', 'CUSTODIAN FEE', 'TAX', 'TOTAL ASSET VALUE', 'FEES']

mergedDF = pd.merge(bg_transaction, ed_transaction, how='outer', on=group_col, indicator=True, suffixes=('_bg', '_ed'))

# mergedDF_Fixed = mergedDF[(mergedDF['Group1'] == 'FIXED INCOME') & (mergedDF['_merge'] == 'both')]
mergedDF_Fixed = mergedDF[(mergedDF['_merge'] == 'both') & (~mergedDF['Group1'].isin(exclude_list))]

# mergedDF_Fixed['MtkAiVal_bg'] = mergedDF_Fixed['HLD_MarketValue_bg'] + mergedDF_Fixed['HLD_AI_bg']
# mergedDF_Fixed['MtkAiVal_ed'] = mergedDF_Fixed['HLD_MarketValue_ed'] + mergedDF_Fixed['HLD_AI_ed']
# mergedDF_Fixed['total_return'] = (mergedDF_Fixed['MtkAiVal_ed']/mergedDF_Fixed['MtkAiVal_bg']) - 1
# mergedDF_Fixed['duration_return'] = -mergedDF_Fixed['HLD_ModDuration_bg'] * (mergedDF_Fixed['HLD_Yield_ed'] - mergedDF_Fixed['HLD_Yield_bg']) * 100
# mergedDF_Fixed['fx_return'] = (mergedDF_Fixed['CurrencyRate_ed']/mergedDF_Fixed['CurrencyRate_bg']) - 1


wb.sheets('Merged').range('A5').value = mergedDF_Fixed.values
wb.sheets('Merged').range('A4').value = mergedDF_Fixed.columns.values
# #%% Pivot
# temp_df_list = temp_df.to_list()
# # temp_df.pivot(columns='HoldDate', values='HLD_MarketValue')



# #%% test pandas manipulate 
# for i in temp_df['PortfolioCode'].unique():
# 	# print(i, "len:", len(i), "len_trim: ", len(i.strip()))
# 	if i in bg_transaction['PortfolioCode'].values:
# 		print(i)



# ================================ Query from ContributionBySecFI ================================
#%% 
import pandas as pd
pd.options.mode.chained_assignment = None  
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import xlwings as xw
import datetime

wb = xw.books('Mockup_port.xlsb')

bg_date = wb.sheets('Main').range("B1").value 
ed_date = wb.sheets('Main').range("D1").value 

conn = psycopg2.connect(user="postgres", 
		password="root", 
		host="10.56.35.166",
		port="5433",
		database="PortAnalytic-2016-17v14")

cursor = conn.cursor(cursor_factory=RealDictCursor)


#%%
ref_port = wb.sheets('Portfolios').range('Q1').value	
ref_bm = wb.sheets('Portfolios').range('Q2').value
port_list = tuple(wb.sheets('Portfolios').range(ref_port).value)
bm_list = tuple(wb.sheets('Portfolios').range(ref_bm).value)
all_list = port_list + bm_list
#%%
cursor.execute('''
	SELECT pa.*, pa_FI.* FROM dbo."PA_ContributionBySecFI" pa_FI
	FULL JOIN dbo."PA_ContributionBySec" pa ON pa."Id" = pa_FI."Id"
	WHERE pa."ParentPortfolioCode" IN %s
	AND pa."ReturnsDate" BETWEEN %s AND %s ORDER BY pa."ParentPortfolioCode" ASC
''', (all_list, bg_date, ed_date))

port_record = pd.DataFrame(cursor.fetchall())
pf_becl = port_record[port_record['ParentPortfolioCode'] == 'PFBECL']

#%% use Group by ContributionType
pivot = pd.pivot_table(pf_becl, values=['ProfitLossPercent', 'BaseWeight'],index=['ReturnsDate','ContributionGroup1', 'ContributionGroup3','ContributionType','PortfolioCode'], aggfunc=np.sum)
wb.sheets('Groupby').range("A1").value = pivot




#%% Group by data 
filter_rec = pf_becl[pf_becl['ReturnsDate'].dt.date == datetime.date(2019, 2, 26)]
wb.sheets('pf_becl').range('A1').value = filter_rec
# table = pd.pivot_table(port_record[port_record['ContributionType'] == 'Fund.FI.Tenor'], values=['DurationPL', 'FXgainloss', 'MarketValue_PER100', 'ProfitLoss',
# 	'OtherPL', 'ProfitLoss', 'ProfitLossPercent', 'ProfitLossPercentPro100',
# 	'RealizedGL', 'RealizedPercent', 'TotalIncome', 'UnrealizedGL', 'YieldPL',
# 	'amortCost_PER100', 'avgCostYield', 'avgHoldingCost_PER100', 'unCatPercent', 'unCategorizedGL'], index=['ParentPortfolioCode', 'ReturnsDate', 'ContributionGroup3'],
# 	aggfunc=np.sum)

# table = pd.pivot_table(port_record, values=['DurationPL', 'FXgainloss', 'MarketValue_PER100', 'ProfitLoss',
# 	'OtherPL', 'ProfitLoss', 'ProfitLossPercent', 'ProfitLossPercentPro100',
# 	'RealizedGL', 'RealizedPercent', 'TotalIncome', 'UnrealizedGL', 'YieldPL',
# 	'amortCost_PER100', 'avgCostYield', 'avgHoldingCost_PER100', 'unCatPercent', 'unCategorizedGL'], index=['ParentPortfolioCode', 'ReturnsDate', 'ContributionGroup3'],
# 	aggfunc=np.sum)


# group_data = port_record.groupby(['ParentPortfolioCode', 'ReturnsDate', 'ContributionGroup3', 'ContributionType']).sum()
# wb.sheets('Merged').range("A1").value = port_record.columns.values
# wb.sheets('Merged').range("A2").value = port_record.values

# wb.sheets('Groupby').range('A1').value = table

# # Define Custom aggregate function

# #%%
# wb.sheets('Merged').range('A1').value = port_record







#%%
