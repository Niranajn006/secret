# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:30:50 2018

@author: s88079
"""

#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt



# For Univariate model
df = pd.read_excel(r'D:\Project files\Data\FX_Historical_data.xlsx',
                   sheet_name='Daily USDTHB',
                   index_col='Date',
                   usecols="A:F")

# For Multivariate model


#creating dataframe
#data = df.sort_index(ascending=True, axis=0)
#new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
#for i in range(0,len(data)):
#    new_data['Date'][i] = data.index[i]
#    new_data['Close'][i] = data['Close'][i]
#
##setting index
#new_data= new_data.index = new_data.Date
#new_data = new_data.drop('Date', axis=1, inplace=True)

new_data = df['Close']

#creating train and test sets
dataset = new_data.values
dataset = dataset.reshape(len(dataset), 1)

pct_train = 0.8
data_train = int(np.round(len(new_data)*pct_train))

train = dataset[:data_train,]
valid = dataset[data_train:,]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=250, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dense(25, activation="sigmoid"))
model.add(LSTM(units=100))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=2)

print(model.summary())

#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))


# Vistualization 
plt_train = new_data[:data_train]
plt_test = new_data[data_train:]

plt.plot(plt_test.values)
plt.plot(closing_price)

#plt_test = pd.concat([plt_test, closing_price.reshape(-1)], axis=1, join='outer', ignore_index=True)
