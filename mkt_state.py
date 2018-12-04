# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 09:29:33 2018

@author: s88079
"""
#%% Import library
import pandas as pd
import numpy as np
from keras.layers import Input, Dense, LSTM, Dropout
from keras.models import Model, Sequential
from keras.preprocessing.sequence import TimeseriesGenerator
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns 

#%% Get Data and generate label and train data
raw_df = pd.read_excel(r'D:\Project files\Data\Mkt timing.xlsx', sheet_name='Level_lag',
                       index_col='Dates', usecols='A:K')

dataset_columns = raw_df.columns.tolist()

# found #N/A in GEMLTL (04/01/2016)
raw_df.dropna(inplace=True)

#%% Select sample data generator  **(Optiona)
rolling_data_set = []
recursive_data_set = []

for i in list(range(2000, 2018)):
    st_year = i
    end_year = i + 6
    if i <= 2012: #<-- loop thoght 2012 where data left 6 year 
        df = raw_df[(raw_df.index.year >=st_year) & (raw_df.index.year <=end_year)]
        rolling_data_set.append(df)
        

for i in list(range(6, 19)):
    st_year = 2000
    end_year = st_year + i
    df = raw_df[(raw_df.index.year >=st_year) & (raw_df.index.year <=end_year)]
    recursive_data_set.append(df)
                

#%% Basic statistic describe
sns.boxplot(data=raw_df)
sns.heatmap(raw_df.corr(), vmax=.3, center=0, square=True, linewidths=.5, cmap='Blues')

#%% At level data  test 1 year
values = raw_df.values   #<-- Change this type of data

diff_values = raw_df.pct_change()

# for loop generate sequence
roll_data = rolling_data_set[1]
test_year = roll_data.index[-1].year
test_data_len = len(roll_data[roll_data.index.year == test_year])

#%% Data Processing 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_dataset = scaler.fit_transform(values)

#%% Generate the sequence of data *** (Optional)
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

# Drop (t) train data use (t-1) forecast ACWI(t)
all_values.drop(all_values.columns[range(len(all_values.columns)-1, 10, -1)], axis=1, inplace=True)


#%% Arrange the data as sequences for training and prediction ***(Optional)
seq_len = 1 # look back or time step

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

# Split into train and test sets

dataX = np.array(dataX)
dataY = np.array(dataY)

#3D [samples, timesteps, features]
train_X, train_y = dataX[n_train_hours:], dataY[n_train_hours:]
test_X, test_y = dataX[:n_train_hours], dataY[:n_train_hours]

#%% Use keras TimeseriesGenerator
def get_y_from_generator(gen):
    '''
    Get all targets y from a TimeseriesGenerator instance.
    '''
    y = None
    for i in range(len(gen)):
        batch_y = gen[i][1]
        if y is None:
            y = batch_y
        else:
            y = np.append(y, batch_y)
    y = y.reshape((-1,1))
    print(y.shape)
    return y


#%% Loop generate forcast each dataset
    
rolling_predict = []

for ele in rolling_data_set:
    test_year = ele.index[-1].year
    test_data_len = len(ele[ele.index.year == test_year])
    df = pd.DataFrame(ele)
    run_model(df)

#df = pd.DataFrame(roll_data)
#%% Define Modl function 
def run_model(df):
    # Use roll_data set 
    dataset_X = df.values[:, :]
    dataset_Y = df.values[:, 0]
    dataset_Y = dataset_Y.reshape(-1, 1)
    
    # Transform data seperate x and y scaler
    scaler_multi = MinMaxScaler(feature_range=(0, 1))
    scaler_multi.fit_transform(dataset_X.reshape(-1, 1))
    #scaler_multi.fit_transform(dataset_Y.reshape(1, -1))
    dataset_X = scaler_multi.transform(dataset_X)
    dataset_Y = scaler_multi.transform(dataset_Y)
    
    
    # split into train and test sets
    train_size = int(len(df) - test_data_len)
    train_X, test_X = dataset_X[:train_size,:], dataset_X[train_size:, :]
    train_y, test_y = dataset_Y[:train_size,:], dataset_Y[train_size:, :]
    
    # set look back period 
    look_back = 5
    
    # Generate Timeseries generator 
    # Stateless lookback equal stride size 
    train_data_gen = TimeseriesGenerator(train_X, train_y,
                                   length=look_back, sampling_rate=1,stride=1,
                                   batch_size=1)
    
    test_data_gen = TimeseriesGenerator(test_X, test_y,
                                   length=look_back, sampling_rate=1,stride=1,
                                   batch_size=1)
    
    # Construct LSTM model 
    model = Sequential()
    model.add(LSTM(10, return_sequences=True, input_shape=(look_back, train_X.shape[1])))
    model.add(Dense(10, activity_regularizer= regularizers.l2(10e-3)))
    model.add(Dropout(0.2))
    model.add(LSTM(5))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    
    # Fit model 
    history = model.fit_generator(train_data_gen, epochs=20).history
    
    # Output model loss
    model.evaluate_generator(test_data_gen)
    
    # Predict 
    trainPredict = model.predict_generator(train_data_gen)
    testPredict = model.predict_generator(test_data_gen)
    
    # Invert scale 
    trainPredict = scaler_multi.inverse_transform(trainPredict)
    testPredict = scaler_multi.inverse_transform(testPredict)
    dataset_Y = scaler_multi.inverse_transform(dataset_Y)
    
    trainY = get_y_from_generator(train_data_gen)
    testY = get_y_from_generator(test_data_gen)
    trainY = scaler_multi.inverse_transform(trainY)
    testY = scaler_multi.inverse_transform(testY)
    
    # Vistualize 
    trainPredictPlot = np.empty_like(dataset_Y)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    
    testPredictPlot = np.empty_like(dataset_Y)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2):len(dataset_Y), :] = testPredict
    
    result = {}
    result['history'] = history
    result['model'] = model
    result['train_predict'] = trainPredictPlot
    result['test_predict'] = testPredictPlot
    return result
    
    #plt.plot(dataset_Y, label='Actual')
    #plt.plot(trainPredictPlot, label='Train Predict')
    #plt.plot(testPredictPlot, label='Test Predict')
    #plt.legend(shadow=True)


#%% Loop generate forcast each dataset
    
rolling_predict = []
recursive_predict = []

for ele in recursive_data_set:
    test_year = ele.index[-1].year
    test_data_len = len(ele[ele.index.year == test_year])
    df = pd.DataFrame(ele)
    result = run_model(df)
    