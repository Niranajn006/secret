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

db = Database('sqlite', r"D:\Project files\Port-Performance\DB\test_db.sqlite", create_db=True)

#db = Database()


class Bond_Info(db.Entity):
    Symbol = PrimaryKey(str)
    SymbolOrder = Optional(str)
    Group = Optional(str)
    GroupOrder = Optional(int)
    GroupName = Optional(str)
    CouponPaymentType = Optional(str)
    Des_PaymentType = Optional(str)
    Par = Optional(float)
    EmbeddedOption = Optional(str)
    InterestRate = Optional(float)
    IssuedDate = Optional(date)
    MaturityDate = Optional(date)
    CurrencyCode = Optional(str)
    IssuerTypeName = Optional(str)
    DistributionType = Optional(str)
    m_t_m__datas = Set('MTM_Data')
    e_o_d__datas = Set('EOD_Data')


class MTM_Data(db.Entity):
    bond__info = Required(Bond_Info)
    MTMDate = Required(date)
    TradeYield = Optional(float)
    QuoteDate = Optional(date, nullable=True)
    MinQuoteYield = Optional(float)
    MaxQuotedYield = Optional(float)
    ModelYield = Optional(float)
    StaticSpread = Optional(float)
    TTM = Optional(Decimal)
    YTM = Optional(float)
    CleanPrice = Optional(float)
    AccruedInterest = Optional(float)
    GrossPrice = Optional(float)
    ModifiedDuration = Optional(float)
    Convexity = Optional(float)
    IndexRatio = Optional(float)
    Outstanding = Optional(float)
    IssueRatingTris = Optional(str)
    IssueRatingFitchth = Optional(str)
    IssueRatingMoody = Optional(str)
    IssueratingRI = Optional(str)
    composite_key(bond__info, MTMDate)


class EOD_Data(db.Entity):
    bond__info = Required(Bond_Info)
    EODDate = Required(date)
    Attribute = Optional(str)
    PrevlastYield = Optional(float)
    PrevlastPrice = Optional(float)
    PrevRemarkFlag = Optional(str)
    PrevWeightAvgYield = Optional(float)
    PrevWeightAvgPrice = Optional(float)
    BestBidYield = Optional(float)
    BestBidPrice = Optional(float)
    AvgBidYield = Optional(float)
    AvgBidPrice = Optional(float)
    BestOfferYield = Optional(float)
    BestOfferPrice = Optional(float)
    WeightAvgOfferYield = Optional(float)
    WeightAvgOfferPrice = Optional(float)
    LastExecYield = Optional(float)
    LastExePrice = Optional(float)
    LastExeGrossValue = Optional(float)
    RemarkFlag = Optional(str)
    WeightAvgYield = Optional(float)
    WeightAvgPrice = Optional(float)
    TotalGrossValue = Optional(float)
    composite_key(bond__info, EODDate)

# db.generate_mapping(create_tables=True)

#%% ============= Define function get data ============= 
def get_bond_info(driver, date_params):
    """
        Get bond infomation from MTM 
    """
    # List All bonds universe from ThaiBMA 
    list_bond_group = ['TB', 'GB', 'SA', 'SOE', 'CORP', 'CP', 'FB', 'USD', 'CNY', 'EUR']
    temp_bond_master = []
    for bg in list_bond_group:
        driver.get('http://www.thaibma.or.th/registeredbond/GetBondOutstandingList/%s'%bg)
        doc = BeautifulSoup(driver.page_source, 'html.parser').get_text()
        raw_bg = json.loads(doc)
        temp_bond_master += raw_bg
    
    # Bond Master data
    bond_master_col = ['Symbol', 'Des_PaymentType', 'IssuedDate', 'MaturityDate', 'CurrencyCode',
    'IssuerTypeName','bg_IssueRatingTris', 'bg_IssueRatingFitchth', 'bg_IssueRatingMoody', 'bg_IssueRatingSP',
    'bg_IssueratingFitch', 'bg_IssueratingRI', 'bg_InterestRate', 'bg_TTM', 'bg_Outstanding']
    
    # login create session
    driver.get('http://www.ibond.thaibma.or.th/EN/Login/Login.aspx?ReturnUrl=%2fEN%2fBondInfo%2fRegisteredBond%2fRegisterSummary.aspx')
    driver.find_element_by_id("txtUserName").send_keys("p1sscba5")
    driver.find_element_by_id('txtPassword').send_keys("mye7he")
    driver.find_element_by_id('btnLogin').click()

    # STN data
    stn_col_name = ['STNDate','IntrumentCode' 'Symbol', 'SymbolOrder', 'RegistrationCode', 'MaturityDate',
    'SettlementDate', 'TTM', 'CouponType', 'PrincipalType', 'Tris', 'Fitch', 'Moody', 'SP', 'FitchInter',
    'RI', 'TradeDate', 'TradePrice', 'QuotedDate', 'QuotedPrice', 'ModelPrice', 'GrossPrice', 'CurrencyCode',
    'Par', 'OutstandingMillion', 'IssuerID', 'IssuerRating', 'DistributionType', 'EmbeddedOption']

    path = 'http://www.ibond.thaibma.or.th/api/stnmarketprice/getdata?asof=%s'%date_params
    driver.get(path)
    doc = BeautifulSoup(driver.page_source, 'html.parser').get_text()
    raw_stn = json.loads(doc)

    # MTM data
    mtm_col_name = ['MTMDate','IntrumentCode' 'Symbol', 'SymbolOrder', 'CurrencyCode', 'RegistrationCode',
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
    eod_col_name = ['EODDate', 'Symbol', 'Attribute', 'Coupon', 'lastDate',
    'PrevlastYield', 'PrevlastPrice', 'PrevRemarkFlag', 'PrevWeightAvgYield', 'PrevWeightAvgPrice',
    'BestBidYield', 'BestBidPrice', 'AvgBidYield', 'AvgBidPrice', 'BestOfferYield', 'BestOfferPrice',
    'WeightAvgOfferYield', 'WeightAvgOfferPrice', 'LastExecYield', 'LastDM', 'LastYTPYTC', 'LastExePrice',
    'LastExeGrossValue', 'RemarkFlag', 'WeightAvgYield', 'WeightAvgPrice', 'TotalGrossValue']

    path = 'http://www.ibond.thaibma.or.th/api/dailytrading/outtrans?asof=%s'%date_params
    driver.get(path) #Second call 
    doc = BeautifulSoup(driver.page_source, 'html.parser').get_text()
    raw_eod = json.loads(doc)

    bond_master = []
    mtm_data = []
    eod_data = []
    stn_data = []

    for i in temp_bond_master:
        ele = {
        'Symbol': i['Symbol'].strip(),
        'Des_PaymentType': i['CouponPaymentType'],
        'IssuedDate': i['IssuedDate'],  #modify date
        'MaturityDate': i['MaturityDate'], #modify date
        'CurrencyCode': i['CurrencyCode'],
        'IssuerTypeName': i['IssuerTypeName'],
        'bg_IssueRatingTris': i['IssueRatingTris'],
        'bg_IssueRatingFitchth': i['IssueRatingFitchth'],
        'bg_IssueRatingMoody': i['IssueRatingMoody'],
        'bg_IssueRatingSP': i['IssueRatingSP'],
        'bg_IssueratingFitch': i['IssueratingFitch'],
        'bg_IssueratingRI': i['IssueratingRI'],
        'bg_InterestRate': i['InterestRate'],
        'bg_TTM': i['TTM'],
        'bg_Outstanding': i['Outstanding']
        }
        bond_master.append(ele)

    bond_master = pd.DataFrame(bond_master)
    # bond_master = bond_master.reindex(columns=bond_master_col)
    bond_master['IssuedDate'] = pd.to_datetime(bond_master['IssuedDate'], format='%Y/%m/%d').dt.date
    bond_master['MaturityDate'] = pd.to_datetime(bond_master['MaturityDate'], format='%Y/%m/%d').dt.date
    # bond_master = bond_master.to_records()

    for i in raw_stn:
        ele = {
            'Asof': date_params, # date modify
            'IntrumentCode': 'SN',
            'Symbol': i['Symbol'].strip(),
            'SymbolOrder': i['SymbolOrder'],
            'RegistrationCode': i['RegistrationCode'],
            'MaturityDate': i['MaturityDate'], #date modify
            'SettlementDate': i['SettlementDate'], #date modify
            'TTM': i['TTM'],
            'CouponType': i['CouponType'],
            'PrincipalType': i['PrincipalType'],
            'Tris': i['Tris'],
            'Fitch': i['Fitch'],
            'Moody': i['Moody'],
            'SP': i['SP'],
            'FitchInter': i['FitchInter'],
            'RI': i['RI'],
            'TradeDate': i['TradeDate'],
            'TradePrice': i['TradePrice'],
            'QuotedDate': i['QuotedDate'], #date modify
            'QuotedPrice': i['QuotedPrice'],
            'ModelPrice': i['ModelPrice'],
            'GrossPrice': i['GrossPrice'],
            'CurrencyCode': i['CurrencyCode'],
            'Par': i['Par'],
            'OutstandingMillion': i['OutstandingMillion'],
            'IssuerRating': i['IssuerRating'],
            'DistributionType': i['DistributionType'],
            'EmbeddedOption': i['EmbeddedOption']
        }
        stn_data.append(ele)
    
    df_stn = pd.DataFrame(stn_data)
    # df_stn = df_stn.reindex(columns=stn_col_name)
    df_stn['Asof'] = pd.to_datetime(df_stn['Asof'], format='%Y/%m/%d').dt.date
    df_stn['MaturityDate'] = pd.to_datetime(df_stn['MaturityDate'], format='%Y/%m/%d').dt.date
    df_stn['SettlementDate'] = pd.to_datetime(df_stn['SettlementDate'], format='%Y/%m/%d').dt.date
    df_stn['QuotedDate'] = pd.to_datetime(df_stn['QuotedDate'], format='%Y/%m/%d').dt.date
    

    for i in raw_mtm:
        if i['Symbol'] in exclude_symbol:
            continue
        ele = {
            'Asof': date_params, # date modify
            'IntrumentCode': 'BN',
            'Symbol': i['Symbol'].strip(),
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
    # df_mtm = df_mtm.reindex(columns=mtm_col_name)
    df_mtm['Asof'] = pd.to_datetime(df_mtm['Asof'], format='%Y/%m/%d').dt.date
    df_mtm['MaturityDate'] = pd.to_datetime(df_mtm['MaturityDate'], format='%Y/%m/%d').dt.date
    df_mtm['SettlementDate'] = pd.to_datetime(df_mtm['SettlementDate'], format='%Y/%m/%d').dt.date
    df_mtm['TradeDate'] = pd.to_datetime(df_mtm['TradeDate'], format='%Y/%m/%d').dt.date
    df_mtm['QuotedDate'] = pd.to_datetime(df_mtm['QuotedDate'], format='%Y/%m/%d').dt.date
    # df_mtm = df_mtm.to_records()

    for i in raw_eod:
        if i['Symbol'] in exclude_symbol:
            continue
        ele = {
            'Asof': date_params, # date modify
            'Symbol': i['Symbol'].strip(),
            'Attribute': i['Attribute'],
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
    #df_eod = df_eod.reindex(columns=eod_col_name)
    df_eod['Asof'] = pd.to_datetime(df_eod['Asof'], format='%Y/%m/%d').dt.date
    df_eod['lastDate'] = pd.to_datetime(df_eod['lastDate'], format='%Y/%m/%d').dt.date
    # df_eod = df_eod.to_records()
    
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
            
            
    return df_stn, df_mtm, df_eod, bond_master

#%% =============  Main Function ============= 
driver = webdriver.Chrome(r'D:\Project files\Data\chromedriver.exe')
stn, mtm, eod, bond_master = get_bond_info(driver, '2019-02-07')

#%% ============= Test Insert Data ============= 

merged = pd.merge(mtm, stn, how='outer', on=['Asof', 'Symbol', 'IntrumentCode', 'CouponType', 'CurrencyCode',
        'DistributionType', 'EmbeddedOption', 'Fitch', 'FitchInter', 'GrossPrice', 'IssuerRating',
        'MaturityDate', 'Moody', 'OutstandingMillion', 'Par', 'PrincipalType', 'QuotedDate', 'RI',
        'RegistrationCode', 'SP', 'SettlementDate', 'SymbolOrder', 'TTM', 'Tris', 'TradeDate'])

final = pd.merge(merged, bond_master, how='outer', on=['Symbol', 'CurrencyCode', 'MaturityDate',])
reindex_col = ['Asof', 'Symbol', 'SymbolOrder', 'IntrumentCode', 'RegistrationCode', 'Group', 'GroupName',
        'GroupOrder', 'Des_PaymentType', 'Par', 'CouponRate', 'CouponType', 'IssuerTypeName',
        'CurrencyCode', 'DistributionType', 'EmbeddedOption', 'MaturityDate', 'IssuedDate',
        'IssuerRating', 'Fitch', 'FitchInter', 'Moody', 'SP', 'RI', 'Tris', 'bg_IssueRatingFitchth',
        'bg_IssueRatingMoody', 'bg_IssueRatingSP', 'bg_IssueRatingTris', 'bg_IssueratingFitch',
        'bg_IssueratingRI', 'bg_InterestRate', 'bg_Outstanding', 'bg_TTM', 'PrincipalType',
        'CleanPrice', 'GrossPrice', 'IndexRatio', 'Convexity', 'AccruedInterest', 'DM',
        'TBP', 'TTM', 'YTM', 'StaticSpread', 'ModifiedDuration', 'OutstandingMillion',
        'TradeDate', 'TradeYield', 'SettlementDate', 'ModelYield', 'ModelPrice', 'QuotedDate',
        'QuotedPrice', 'MaxQuotedYield', 'MinQuotedYield']
final = final.reindex(columns=reindex_col)


final.to_excel(r'D:\Project files\Data\final.xlsx')

# master_df = pd.merge(mtm, stn, how='outer', on=['Symbol', 'InstrumentCode'])


# merged.to_excel(r'D:\Project files\Data\raw_merged.xlsx')
# db.bind('sqlite', r"D:\Project files\Port-Performance\DB\test_dbs6.sqlite", create_db=True)
# db.generate_mapping(create_tables=True)
#%% DB session
with db_session:
    
    #Mark to Market file
    for i in mtm:
        # 1) Insert Bond info from MTM data
        # 2) Mapping MTM data 
        # 3) Record EOD data 
        if not Bond_Info.exists(Symbol=i['Symbol']):
            bond_record = Bond_Info(Symbol=i['Symbol'], 
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
        
        
        key = MTM_Data.get(bond__info=i['Symbol'], Asof=i['Asof'])
        if key is None:
        # if not MTM_Data.exists(bond__info=(Bond_Info.get(Symbol=i['Symbol'])):
            if not pd.isnull(i['QuotedDate']):
                mtm_record = MTM_Data(bond__info=(Bond_Info.get(Symbol=i['Symbol'])),
                Asof = i['Asof'],
                TradeYield = i['TradeYield'],
                QuoteDate = i['QuotedDate'],
                MinQuoteYield = i['MinQuotedYield'],
                MaxQuotedYield = i['MaxQuotedYield'],
                ModelYield = i['ModelYield'],
                StaticSpread = i['StaticSpread'],
                YTM = i['YTM'],
                CleanPrice = i['CleanPrice'],
                AccruedInterest= i['AccruedInterest'],
                GrossPrice = i['GrossPrice'],
                ModifiedDuration = i['ModifiedDuration'],
                Convexity = i['Convexity'],
                IndexRatio = i['IndexRatio'])
            else:
                mtm_record = MTM_Data(bond__info=(Bond_Info.get(Symbol=i['Symbol'])),
                Asof = i['Asof'],
                TradeYield = i['TradeYield'],
                QuoteDate = None,
                MinQuoteYield = i['MinQuotedYield'],
                MaxQuotedYield = i['MaxQuotedYield'],
                ModelYield = i['ModelYield'],
                StaticSpread = i['StaticSpread'],
                YTM = i['YTM'],
                CleanPrice = i['CleanPrice'],
                AccruedInterest= i['AccruedInterest'],
                GrossPrice = i['GrossPrice'],
                ModifiedDuration = i['ModifiedDuration'],
                Convexity = i['Convexity'],
                IndexRatio = i['IndexRatio'])
db.disconnect()
