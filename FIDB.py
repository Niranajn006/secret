#%% ============= Import Libs ============= 
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup
import pandas as pd 
import json
from datetime import date
from decimal import Decimal
from pony.orm import *

#%% ============= Mapping ORM ============= 

db = Database('sqlite', r"D:\Project files\Port-Performance\DB\test_dbs1.sqlite", create_db=True)

#db = Database()


class Bond_Info(db.Entity):
    Symbol = PrimaryKey(str)
    SymbolOrder = Optional(str)
    Group = Optional(str)
    GroupOrder = Optional(int)
    GroupName = Optional(str)
    IssueRatingTris = Optional(str)
    IssueRatingFitchth = Optional(str)
    IssueRatingMoody = Optional(str)
    IssueratingRI = Optional(str)
    CouponPaymentType = Optional(str)
    Par = Optional(int)
    EmbeddedOption = Optional(str)
    InterestRate = Optional(Decimal)
    MaturityDate = Optional(date)
    IssuedDate = Optional(date)
    TTM = Optional(Decimal)
    Outstanding = Optional(float)
    CurrencyCode = Optional(str)
    DistributionType = Optional(str)
    m_t_m__datas = Set('MTM_Data')
    e_o_d__datas = Set('EOD_Data')


class MTM_Data(db.Entity):
    bond__info = Required(Bond_Info)
    Asof = Required(date)
    TradeYield = Optional(Decimal)
    QuoteDate = Optional(date)
    MinQuoteYield = Optional(Decimal)
    MaxQuotedYield = Optional(Decimal)
    ModelYield = Optional(Decimal)
    StaticSpread = Optional(Decimal)
    YTM = Optional(Decimal)
    CleanPrice = Optional(float)
    AccruedInterest = Optional(Decimal)
    GrossPrice = Optional(float)
    ModifiedDuration = Optional(Decimal)
    Convexity = Optional(Decimal)
    IndexRatio = Optional(Decimal)
    composite_key(bond__info, Asof)


class EOD_Data(db.Entity):
    bond__info = Required(Bond_Info)
    Asof = Required(date)
    Attribute = Optional(str)
    PrevlastYield = Optional(Decimal)
    PrevlastPrice = Optional(float)
    PrevRemarkFlag = Optional(str)
    PrevWeightAvgYield = Optional(Decimal)
    PrevWeightAvgPrice = Optional(float)
    BestBidYield = Optional(Decimal)
    BestBidPrice = Optional(float)
    AvgBidYield = Optional(Decimal)
    AvgBidPrice = Optional(float)
    BestOfferYield = Optional(Decimal)
    BestOfferPrice = Optional(float)
    WeightAvgOfferYield = Optional(Decimal)
    WeightAvgOfferPrice = Optional(float)
    LastExecYield = Optional(float)
    LastExePrice = Optional(float)
    LastExeGrossValue = Optional(float)
    RemarkFlag = Optional(str)
    WeightAvgYield = Optional(Decimal)
    WeightAvgPrice = Optional(float)
    TotalGrossValue = Optional(float)
    composite_key(bond__info, Asof)

#db.generate_mapping()

#%% ============= Define function get data ============= 
def get_bond_info(driver, date_params):
    """
        Get bond infomation from MTM 
    """
    # List All bonds universe from ThaiBMA 

    

    # Alway login create session
    driver.get('http://www.ibond.thaibma.or.th/EN/Login/Login.aspx?ReturnUrl=%2fEN%2fBondInfo%2fRegisteredBond%2fRegisterSummary.aspx')
    driver.find_element_by_id("txtUserName").send_keys("p1sscba5")
    driver.find_element_by_id('txtPassword').send_keys("mye7he")
    driver.find_element_by_id('btnLogin').click()

    # MTM data
    mtm_col_name = ['Asof', 'Symbol', 'SymbolOrder', 'CurrencyCode', 'RegistrationCode',
    'TBP', 'MaturityDate', 'SettlementDate', 'TTM', 'CouponRate', 'CouponType', 'PrincipalType',
    'Group', 'GroupOrder', 'GroupName', 'Tris', 'Fitch', 'Moody', 'SP', 'FitchInter', 'RI',
    'TradeDate', 'TradeYield', 'QuotedDate', 'MinQuotedYield', 'MaxQuotedYield', 'ModelYield',
    'StaticSpread', 'YTM', 'DM', 'CleanPrice', 'AccruedInterest', 'GrossPrice', 'ModifiedDuration',
    'Convexity', 'IndexRatio', 'Par', 'OutstandingMillion', 'IssuerRating', 'DistributionType',
    'EmbeddedOption']

    exclude_symbol = ['1 Government Debt Securities', '1.1 Treasury Bill', '1.2 Government Bond', '1.3 State Agency Bond',
    '1.4 State Own Enterprise Bond', '1.5 Government Promissory Note', '2 Corporate Bond', '2.1 Long-term Corporate Bond',
    '2.2 Commercial Paper', '3 Foreign Bond']

    path = 'http://www.ibond.thaibma.or.th/api/marketyield/getdata?asof=%s&group='%date_params
    driver.get(path) #Second call 
    doc = BeautifulSoup(driver.page_source, 'html.parser').get_text()
    raw_mtm = json.loads(doc)

    # EOD data
    eod_col_name = ['Asof', 'Symbol', 'MaturityDate', 'Attribute', 'TTM', 'Coupon', 'lastDate',
    'PrevlastYield', 'PrevlastPrice', 'PrevRemarkFlag', 'PrevWeightAvgYield', 'PrevWeightAvgPrice',
    'BestBidYield', 'BestBidPrice', 'AvgBidYield', 'AvgBidPrice', 'BestOfferYield', 'BestOfferPrice',
    'WeightAvgOfferYield', 'WeightAvgOfferPrice', 'LastExecYield', 'LastDM', 'LastYTPYTC', 'LastExePrice',
    'LastExeGrossValue', 'RemarkFlag', 'WeightAvgYield', 'WeightAvgPrice', 'TotalGrossValue']

    path = 'http://www.ibond.thaibma.or.th/api/dailytrading/outtrans?asof=%s'%date_params
    driver.get(path) #Second call 
    doc = BeautifulSoup(driver.page_source, 'html.parser').get_text()
    raw_eod = json.loads(doc)

    mtm_data = []
    eod_data = []
    
    for i in raw_mtm:
        if i['Symbol'] in exclude_symbol:
            continue
        ele = {
            'Asof': date_params, # date modify
            'Symbol': i['Symbol'],
            'SymbolOrder': i['SymbolOrder'],
            'CurrencyCode': i['CurrencyCode'],
            'RegistrationCode': i['RegistrationCode'],
            'TBP': i['TBP'],
            'MaturityDate': i['MaturityDate'],  #date modify
            'SettlementDate': i['SettlementDate'], #date modify
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
            'TradeDate': i['TradeDate'], #date modify
            'TradeYield': i['TradeYield'],
            'QuotedDate': i['QuotedDate'], #date modify
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
            'EmbeddedOption': i['EmbeddedOption']
        }
        mtm_data.append(ele)
    
    df_mtm = pd.DataFrame(mtm_data)
    df_mtm = df_mtm.reindex(columns=mtm_col_name)
    df_mtm['Asof'] = pd.to_datetime(df_mtm['Asof'], format='%Y/%m/%d').dt.date
    df_mtm['MaturityDate'] = pd.to_datetime(df_mtm['MaturityDate'], format='%Y/%m/%d').dt.date
    df_mtm['SettlementDate'] = pd.to_datetime(df_mtm['SettlementDate'], format='%Y/%m/%d').dt.date
    df_mtm['TradeDate'] = pd.to_datetime(df_mtm['TradeDate'], format='%Y/%m/%d').dt.date
    df_mtm['QuotedDate'] = pd.to_datetime(df_mtm['QuotedDate'], format='%Y/%m/%d').dt.date
    df_mtm = df_mtm.set_index('Asof')
    df_mtm = df_mtm.to_records()

    for i in raw_eod:
        if i['Symbol'] in exclude_symbol:
            continue
        ele = {
            'Asof': date_params, # date modify
            'Symbol': i['Symbol'],
            'MaturityDate': i['Maturity'], # date modify
            'Attribute': i['Attribute'],
            'TTM': i['TTM'],
            'Coupon': i['Coupon'],
            'lastDate': i['lastDate'], # date modify
            'PrevlastYield': i['PrevlastYield'],
            'PrevlastPrice': i['PrevlastPrice'],
            'PrevRemarkFlag': i['PrevRemarkFlag'],
            'PrevWeightAvgYield': i['PrevWeightAvgYield'],
            'PrevWeightAvgPrice': i['PrevWeightAvgPrice'],
            'BestBidYield': i['BestBidYield'],
            'BestBidPrice': i['BestBidPrice'],
            'AvgBidYield': i['AvgBidYield'],
            'AvgBidPrice': i['AvgBidPrice'],
            'BestOfferYield': i['BestOfferYield'],
            'BestOfferPrice': i['BestOfferPrice'],
            'WeightAvgOfferYield': i['WeightAvgOfferYield'],
            'WeightAvgOfferPrice': i['WeightAvgOfferPrice'],
            'LastExecYield': i['LastExecYield'],
            'LastDM': i['LastDM'],
            'LastYTPYTC': i['LastYTPYTC'],
            'LastExePrice': i['LastExePrice'],
            'LastExeGrossValue': i['LastExeGrossValue'],
            'RemarkFlag': i['RemarkFlag'],
            'WeightAvgYield': i['WeightAvgYield'],
            'WeightAvgPrice': i['WeightAvgPrice'],
            'TotalGrossValue': i['TotalGrossValue']
        }   
        eod_data.append(ele)
    
    df_eod = pd.DataFrame(eod_data)
    df_eod = df_eod.reindex(columns=eod_col_name)
    df_eod['Asof'] = pd.to_datetime(df_eod['Asof'], format='%Y/%m/%d').dt.date
    df_eod['MaturityDate'] = pd.to_datetime(df_eod['MaturityDate'], format='%Y/%m/%d').dt.date
    df_eod['lastDate'] = pd.to_datetime(df_eod['lastDate'], format='%Y/%m/%d').dt.date
    df_eod = df_eod.set_index('Asof')
    df_eod = df_eod.to_records()
    
    driver.close()

    # generate mapping session 
    # df.bind('sqlite', r"D:\Project files\Port-Performance\DB\FixedIncome.sqlite", create_db=True)
    # db.generate_mapping(create_tables=True)

    # with db_session:
    #     for i in df_mtm:
    #         # 1) Insert Bond info from MTM data
    #         # 2) Mapping MTM data 
    #         # 3) Record EOD data 
    #         if not Bond_Info.exists(Symbol=i['Symbol']):
    #             Bond_Info(Symbol=i['Symbol'], SymbolOrder=i['SymbolOrder'], Group=i['Group'], GroupOrder=i['GroupOrder'], GroupName=i['GroupName'],
    #             IssueRatingTris=i['IssueRatingTris'], IssueRatingFitchth=i['IssueRatingFitchth'], IssueRatingMoody=i['IssueRatingMoody'],
    #             IssueratingRI=i['IssueratingRI'], CouponPaymentType=i['CouponPaymentType'], EmbeddedOption=i['EmbeddedOption'],
    #             InterestRate=i['InterestRate'], MaturityDate=i['MaturityDate'], IssuedDate=i['IssuedDate'], TTM=['TTM'],
    #             TotalGrossValue=i['TotalGrossValue'], CurrencyCode=i['CurrencyCode'])
            
    #         # Gen data to MTM table 
    #         symbols_list = select(n for n in Bond_Info)[:]
            
            
    return df_mtm, df_eod

#%% =============  Main Function ============= 
driver = webdriver.Chrome(r'D:\Project files\Data\chromedriver.exe')
mtm, eod = get_bond_info(driver, '2019-02-04')

#%% =============  Insert Data ============= 

#db.database('sqlite', r"D:\Project files\Port-Performance\DB\test_dbs.sqlite", create_db=True)
db.generate_mapping(create_tables=True)

with db_session:
    for i in mtm:
        # 1) Insert Bond info from MTM data
        # 2) Mapping MTM data 
        # 3) Record EOD data 
        if not Bond_Info.exists(Symbol=i['Symbol']):
            Bond_Info(Symbol=i['Symbol'], 
            SymbolOrder=i['SymbolOrder'], 
            Group=i['Group'], 
            GroupOrder=i['GroupOrder'],
            GroupName=i['GroupName'],
            IssueRatingTris=i['Tris'],
            IssueRatingFitchth=i['Fitch'], 
            IssueRatingMoody=i['Moody'],
            IssueratingRI=i['RI'], 
            CouponPaymentType=i['CouponType'], 
            EmbeddedOption=i['EmbeddedOption'],
            InterestRate=i['CouponRate'], 
            Par=i['Par'],
            MaturityDate=i['MaturityDate'], 
            IssuedDate=i['MaturityDate'],
            TTM=i['TTM'],
            Outstanding=i['OutstandingMillion'], 
            CurrencyCode=i['CurrencyCode'], 
            DistributionType=i['DistributionType'])
        
        # Gen data to MTM table 
        symbols_list = select(n for n in Bond_Info)[:]

#%%
db.disconnect()