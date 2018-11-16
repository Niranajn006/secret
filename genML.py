# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 16:42:38 2018

@author: s88079
"""

#'from sklearn.model_selection import TimeSeriesSplit # for recursive window 
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# TimeSeriesSplit set window as recursive form create custom one 

def custom_rolling_win(df, st_index, win_size):
    if len(df[st_index:st_index + win_size]) < win_size:
        pass #return somethin terminate this function
    else:
        return df[st_index:st_index + win_size]
    
# ================================= Main Function ==========================
if __name__ == '__main__':
    file_path = r'D:\Project files\input_ml.csv'
    df = pd.read_csv(file_path, index_col='Date')
    df = df.fillna(0) # Replace NaN value with 0
    df['Signal'] = df['Signal'].astype(int) # convert to int object 
    
    # test gen ml model with python 
    y = df['Signal']
    X = df.drop('Signal', axis=1)
    X = preprocessing.maxabs_scale(X)   #Scale each feature to the [-1, 1] range
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=0,
                                                        shuffle=False)
    
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test) #Test model with test data
    
    
    # set role and normalize 
    #pipeline = make_pipeline(preprocessing.Normalizer(),
    #                         LogisticRegression())
    
    
    
    #df = df.set_index(pd.to_datetime(df['Date'], format='%d-%m-%Y'))
    #df = df.drop(['Date'], axis=1)
    