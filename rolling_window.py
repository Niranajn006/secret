# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 14:25:47 2018

@author: s88079
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime



def custom_rolling_window(df, window_size, st_date):

    #next_window = df.index[st_date + datetime.timedelta(days=window_size)]
    
    next_window = df.index[int((st_date + datetime.timedelta(days=window_size)).date)]
    
    if next_window in df.index:
        return df.loc[(df.index >= st_date) & (df.index < st_date +
                      datetime.timedelta(days=window_size))] 
    else:
        return df.loc[(df.index >= st_date) & (df.index <= df.index[-1])]
#    try:
#        next_window = df.index[st_date + datetime.timedelta(days=window_size)]
#    except IndexError:
#        next_window = df.index[-1]
        
        
if __name__ == '__main__':
    
    df = pd.read_excel(r'D:\Project files\FX_Historical_data.xlsx',
                   sheet_name='Daily USDTHB',
                   index_col='Date',parse_dates=True)

    #data = df.rolling(window=100) #This return rolling object 
    
    # Define rolling windows function 
#    rolling_date = 200
#    rolling_win = df.loc[(df.index >= df.index[0]) & (df.index <= df.index[0] +
#                         datetime.timedelta(days=rolling_date))]
    
    # use time datetime.timedelta(days)
    
    # Vistualization 
#    f, (ax1, ax2) = plt.subplots(2, 1)
#    ax1.plot(df['Close'])
#    ax1.set_title('USDTHB 2008-2018')
#    ax2.plot(rolling_win['Close'])
#    ax2.set_title('USDTHB 2008')
    
    test = custom_rolling_window(df, 200, df.index[1])
    