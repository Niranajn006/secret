#%% Initial
from pony.orm import *
from decimal import Decimal
from datetime import datetime

db = Database("sqlite", r"D:\Project files\Port-Performance\DB\Mark2Markets.sqlite", create_db=True)

from datetime import datetime
from decimal import Decimal
from pony.orm import *


db = Database()


class Bond(db.Entity):
    id = PrimaryKey(int, auto=True)
    mtm_date = Optional(datetime)
    lt_date = Optional(datetime)
    lt_exe_yield = Optional(Decimal)
    lq_date = Optional(datetime)
    lq_avg_yield = Optional(Decimal)
    lq_max_yield = Optional(Decimal)
    lq_min_yield = Optional(Decimal)
    spread = Optional(Decimal)
    mkt_yield = Optional(Decimal)
    clean_price = Optional(Decimal)
    outstanding = Optional(Decimal)
    ttm = Optional(Decimal)
    dtm = Optional(Decimal)
    bonds_info = Required('BondsInfo')


class BondsInfo(db.Entity):
    bonds_id = Set(Bond)
    issuer_rating = Optional(str)
    par = Required(int)
    maturity_date = Optional(datetime)
    payment = Optional(str)
    coupon = Required(Decimal)
    coupon_type = Optional(str)
    ratings = Set('Rating')
    currencys = Set('Currency')
    payments = Set('Payment')
    distribution = Optional(str)


class Rating(db.Entity):
    bs = Set(BondsInfo)
    rating_agent = Optional(str)
    rating_score = Optional(str)


class Payment(db.Entity):
    bs = Set(BondsInfo)
    payment_type = Optional(str)


class Currency(db.Entity):
    bs = Set(BondsInfo)
    currency = Optional(str)

db.generate_mapping(create_tables=True)

#%% Generate simple data 


 
#%% test queries
select(c.currency for c in Currency)[:]



#%% Test read mtmExcel
import pandas as pd

path = r'D:\Project files\Port-Performance\DB\Mark2Market_3001219.xlsx'



#%% Fixed Proxy
# import requests
# import json

# username = "s88079"
# password = "Openmenow20"

# proxies = {"http":"http://%s:%s@proxy14.scb.co.th:8080/"%(username, password),
#            "https":"http://%s:%s@proxy14.scb.co.th:8080/"%(username, password)}

# payload = {'asof':'2019-01-15'}

# s = requests.get('http://www.ibond.thaibma.or.th/api/marketyield/getdata',
#      params=payload, proxies=proxies)


#%% Test Selenium webdrivers
#================= Get MarktoMarket =====================
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup
import pandas as pd 
from datetime import datetime

driver = webdriver.Chrome(r'D:\Project files\Data\chromedriver.exe')

driver.get('http://www.ibond.thaibma.or.th/EN/Login/Login.aspx?ReturnUrl=%2fEN%2fBondInfo%2fRegisteredBond%2fRegisterSummary.aspx')
text_input = driver.find_element_by_id("txtUserName").send_keys("p1sscba5")
pwd_input = driver.find_element_by_id('txtPassword').send_keys("mye7he")
log_in = driver.find_element_by_id('btnLogin').click()
date_params = '2019-01-29'
path = 'http://www.ibond.thaibma.or.th/api/marketyield/getdata?asof=%s&group='%date_params

driver.get(path)

# Instead of using requests.get, we just look at .page_source of the driver
doc = BeautifulSoup(driver.page_source, 'html.parser')
raw = doc.get_text()
j = json.loads(raw)

#%% loop gen dataframe
column_list = ['Asof', 'IssueID', 'Symbol', 'SymbolOrder', 'CurrencyCode', 'RegistrationCode',
'TBP', 'MaturityDate', 'SettlementDate', 'TTM', 'CouponRate', 'CouponType', 'PrincipalType',
'Group', 'GroupOrder', 'GroupName', 'Tris', 'Fitch', 'Moody', 'SP', 'FitchInter', 'RI',
'TradeDate', 'TradeYield', 'QuotedDate', 'MinQuotedYield', 'MaxQuotedYield', 'ModelYield',
'StaticSpread', 'YTM', 'DM', 'CleanPrice', 'AccruedInterest', 'GrossPrice', 'ModifiedDuration',
'Convexity', 'IndexRatio', 'Par', 'OutstandingMillion', 'IssuerRating', 'DistributionType',
'DisplayDate', 'EmbeddedOption']
# date format datetime.strptime(j[1]['Asof'][:10], '%Y-%m-%d')

data = []

for i in j:
    ele = {
        'Asof': i['Asof'],
        'IssueID': i['IssuerID'],
        'Symbol': i['Symbol'],
        'SymbolOrder': i['SymbolOrder'],
        'CurrencyCode': i['CurrencyCode'],
        'RegistrationCode': i['RegistrationCode'],
        'TBP': i['TBP'],
        'MaturityDate': i['MaturityDate'],
        'SettlementDate': i['SettlementDate'],
        'TTM': i['TTM'],
        'CouponRate': i['CouponRate'],
        'CouponType': i['CouponType'],
        'PrincipalType': i['PrincipalType'],
        'Group': i['Group'],
        'GroupOrder': i['GroupOrder'],
        'GroupName': i['GroupName'],
        'Tris': i['Tris'],
        'Fitch': i['Fitch'],
        'Moody': i['Moody'],
        'SP': i['SP'],
        'FitchInter': i['FitchInter'],
        'RI': i['RI'],
        'TradeDate': i['TradeDate'],
        'TradeYield': i['TradeYield'],
        'QuotedDate': i['QuotedDate'],
        'QuotedYield': i['QuotedYield'],
        'MinQuotedYield': i['MinQuotedYield'],
        'MaxQuotedYield': i['MaxQuotedYield'],
        'ModelYield': i['ModelYield'],
        'StaticSpread': i['StaticSpread'],
        'YTM': i['YTM'],
        'DM': i['DM'],
        'CleanPrice': i['CleanPrice'],
        'AccruedInterest': i['AccruedInterest'],
        'GrossPrice': i['GrossPrice'],
        'ModifiedDuration': i['ModifiedDuration'],
        'Convexity': i['Convexity'],
        'IndexRatio': i['IndexRatio'],
        'Par': i['Par'],
        'OutstandingMillion': i['OutstandingMillion'],
        'IssuerRating': i['IssuerRating'],
        'DistributionType': i['DistributionType'],
        'DisplayDate': i['DisplayDate'],
        'EmbeddedOption': i['EmbeddedOption']
    }
    data.append(ele)
#%% Make dataframe
df = pd.DataFrame(data)
df = df.reindex(columns=column_list)
df['Asof', ] = pd.to_datetime(df['Asof'])
df['MaturityDate'] = pd.to_datetime(df['MaturityDate'])
df['SettlementDate'] = pd.to_datetime(df['SettlementDate'])
df['TradeDate'] = pd.to_datetime(df['TradeDate'])
df['QuotedDate'] = pd.to_datetime(df['QuotedDate'])
df['DisplayDate'] = pd.to_datetime(df['DisplayDate'])
df = df.set_index('Asof')
df.head()

#%% 
df.to_excel(r'D:\Project files\Data\output.xlsx')
driver.close()