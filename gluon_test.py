# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 08:22:52 2018

@author: User
"""


import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
plt.style.use('ggplot')

#%%Generate data set
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')

org_col_names=["No", "year","month", "day", "hour", "pm2.5", "DEWP","TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]
col_names = ['pollution', 'dew', 'temp', 'pressure', 'w_dir', 'w_speed', 'snow', 'rain']         
dataset = pd.read_csv(r'D:\Projects\Data\PRSA_data_2010.1.1-2014.12.31.csv',  
                    index_col=0,
                    date_parser=parse,
                    parse_dates=[['year', 'month', 'day', 'hour']])

# Data cleansing
dataset.drop('No', axis=1, inplace=True)
dataset.columns = col_names
dataset['pollution'].fillna(0, inplace=True)
dataset = dataset[24:] # drop the first day
#print(d!ataset.head(5))
dataset.to_csv('pollution.csv') # save new CSV

df = pd.read_csv('pollution.csv', header=0, index_col=0)
dataset_columns = df.columns.tolist() #keep column name 

#df.boxplot() # vistualize basic statistic data 
sns.boxplot(data=df)

plt.figure()
cor_cols = ['pollution', 'wnd_spd', 'rain', 'snow', 'temp']
sns.heatmap(df.loc[:, cor_cols].corr(), vmax=.3, center=0, square=True, linewidths=.5, cmap='Blues')

#%% Data Processing 
values = df.values
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4]) # tranform w_dir 
values = values.astype('float64')
values[:, 4]

scaler = MinMaxScaler(feature_range=(0,1))
scaled_dataset = scaler.fit_transform(values)

#%%Generate the sequence ======================
df = pd.DataFrame(scaled_dataset)
cols= []
col_names = []

n_in = 1 
n_out = 1

n_vars = scaled_dataset.shape[1]
# input sequence (t, t+1, ... t+n)
for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    col_names += [('%s(t-%s)' % (dataset_columns[j], i)) for j in range(n_vars)]
    
# forecast pollution values
cols.append(df.shift(0))
col_names += [('%s(t)'%dataset_columns[j]) for j in range(n_vars)]

all_values = pd.concat(cols, axis=1)
all_values.columns = col_names
all_values.dropna(inplace=True)

# Only forecasting one variable(pollution) next day, so drop the rest
all_values.drop(all_values.columns[range(9, 16)], axis=1, inplace=True)

#%% Arrange the data as sequences for trainning and prediction 
seq_len = 4 # sequence length or time step
print("Dataset Shape: ", all_values.values.shape)

X = all_values.values[:, :-1] #extract all column except last column for train variable
y = all_values.values[:, -1] #extract last column for target variable

dataX = []
dataY = []
for i in range(0, len(y) - seq_len):
    _x = X[i: i+seq_len]
    _y = y[i: i+seq_len]
    _y = _y[-1] #pick last one as the forecast target
    dataX.append(_x)
    dataY.append(_y)
 
#%% Keep the original X and y for testing 
n_train_hours = 365 * 24 #save origin for 1 year 
o_test_X, o_test_y = X[:n_train_hours], y[:n_train_hours]
o_test_X.shape, o_test_y.shape

#%% Split into train and test sets
dataX = np.array(dataX)
dataY = np.array(dataY)
#3D [samples, timesteps, features]
train_X, train_y = dataX[n_train_hours:], dataY[n_train_hours:]
test_X, test_y = dataX[:n_train_hours], dataY[:n_train_hours]

#%% Build Keras model 
from keras import Sequential
from keras.layers import LSTM, Dense
# Construct model
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# Fit network
history = model.fit(train_X, train_y, epochs=5,
                    batch_size=72, validation_data=(test_X, test_y),
                    verbose=2, shuffle=False)

# Plot model result
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
#%% Build MXnet Model 
import mxnet as mx
import logging
batch_size = 72
train_iter = mx.io.NDArrayIter(data=train_X, label=train_y,
                               data_name='data', label_name='target',
                               batch_size=batch_size, shuffle=False)
test_iter = mx.io.NDArrayIter(data=test_X, label=test_y,
                              data_name='data', label_name='target',
                              batch_size=batch_size)

logging.getLogger().setLevel(logging.INFO)

ctx = [mx.cpu(i) for i in range(1)]

# Define the LSTM Neural Network
num_epochs = 25

# Note that when  unrolling, if 'merge_outputs' is set to True, the 'outputs' is
# merged into a single symbol 
# In the layout, 'N' represents batch size, 'T' represents sequence length,
# and 'C' represents the number of dimensions in hidden states.

data = mx.sym.var('data') # Shape: (N, T, C)
target = mx.sym.var('target')
data = mx.sym.transpose(data, axes=(1, 0, 2)) # Shape: (T, N, C)

if isinstance(ctx, list):
    c_ctx = ctx[0]
else:
    c_ctx = ctx
    
if c_ctx.device_type == 'cpu':
    lstm1 = mx.rnn.LSTMCell(num_hidden=5, prefix='lstm1_')
    lstm2 = mx.rnn.LSTMCell(num_hidden=10, prefix='lstm2_')
else:
    #FusedRNNCell
    lstm1 = mx.rnn.FusedRNNCell(num_hidden=5, mode='lstm', prefix='lstm1_')
    lstm2 = mx.rnn.FusedRNNCell(num_hidden=10, mode='lstm', prefix='lstm2_')
    
L1, L1_states = lstm1.unroll(length=seq_len, inputs=data,
                             merge_outputs=True,
                             layout="TNC")  # Shape: (T, N, 5)

L1 = mx.sym.Dropout(L1, p=0.2) # Shape: (T, N, 5)
L2, L2_states = lstm2.unroll(length=seq_len, inputs=L1,
                             merge_outputs=True,
                             layout="TNC") # Shape: (T, N, 10)

L2 = mx.sym.reshape(L2_states[0], shape=(-1, 0), reverse=True) # Shape: (T*N, 10)
pred = mx.sym.FullyConnected(L2, num_hidden=1, name="pred")
pred = mx.sym.LinearRegressionOutput(data=pred, label=target)

model = mx.mod.Module(symbol=pred, data_names=['data'],
                      label_names=['target'], context=ctx)

model.fit(train_data=train_iter, eval_data=test_iter,
          initializer=mx.init.Xavier(rnd_type='gaussian', magnitude=1),
          optimizer='adam', optimizer_params={'learning_rate':1e-3},
          batch_end_callback=mx.callback.Speedometer(batch_size, 100),
          eval_metric='mse', num_epoch=num_epochs)

#%% Let make prediction 
import math
from sklearn.metrics import mean_squared_error

# Prediction
yhat = model.predict(test_iter).asnumpy()
#print(np.mean((yhat - test_y)))

#print yhat.shape, test_X.shape, test_X[:, 1:].shape

p_test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
p_test_X.shape

inv_yhat = np.concatenate((yhat, p_test_X[:, 1:]), axis=1)
#print inv_yhat.shape

inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# scale back
scaled_test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((scaled_test_y, p_test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# calculate RMSE
rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
#print('Test RMSE: %.3f' % rmse)

