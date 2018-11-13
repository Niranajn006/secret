from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder # Encode label to catagory
from sklearn.metrics import mean_squared_error
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def scale_data(df_train, df_test, trans_class=False):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = df_train.values
    test = df_test.values
    train = train.reshape(train.shape[0], train.shape[1])
    test = test.reshape(test.shape[0], test.shape[1])
    if trans_class:
        train[:, -1] = to_categorical(train[:, -1])
        test[:, -1] = to_categorical(test[:, -1])
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.fit_transform(test)
    return scaler, train_scaled, test_scaled

def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


def fit_lstm(train, batch_size, nb_epoch, neurons):
    '''
    stateful model: when run in “stateful” mode, we can often get high 
    accuracy results by leveraging the autocorrelations present in the time series.
    '''
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1],
                                               X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        model.reset_states()
    return model

def forecast_lstm(model, batch_size, X, trans_class=False):
    X = X.reshape(1, 1, len(X))
    if trans_class:
        yhat = model.predict_classes(X, batch_size=batch_size)
    else:
        yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]



if __name__ ==  '__main__':
    
# =============================== Initail Model ===============================
    df = pd.read_excel(r'D:\Project files\FXProjects\Mkt timing.xlsx',
                       sheet_name='Level_lag',
                       index_col='Dates',
                       usecols="A:J")
    
    # Train model with 10 year data and test look ahead 1 year rolling 1 step
    df_train = df[df.index.year <= 2010]
    df_test = df[df.index.year == 2011]
    
    # Transform train/test data
    scaler, train_scaled, test_scaled = scale_data(df_train, df_test, trans_class=False)
    
    # Fit model 
    lstm_model = fit_lstm(train_scaled, 1, 50, 50)
    # Forecast the entire training dataset to build up state for forecasting
    #train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    
    # For regression problem 
    #lstm_model.predict(train_reshaped, batch_size=1)
    # For classification problem
    #lstm_model.predict_classes(train_reshaped, batch_size=1)
    
    # walk-foreward validation on the test data 
    predictions = list()
    for i in range(len(test_scaled)):
        # make one-step ahead forecast 
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forecast_lstm(lstm_model, 1, X, trans_class=True)
        # invert scaled 
        yhat = invert_scale(scaler, X, yhat)
        # store forecast 
        predictions.append(yhat)
        # retrive y_test 
        expected = df.values[len(df_train) + i + 1, -1]
        print('Date=%d, Predicted=%d, Expected=%d' %(i+1, yhat, expected))
    
    
    
    # Vistualize forecast 
    plt.plot(df.iloc[len(df_train):, 0])
    
    

    




# =========================== Use Function generate ==========================

##values[:, 16] = to_categorical(values[:, 16])
#
## Normalize features
#scaler = MinMaxScaler(feature_range=(-1, 1))
#scaled = scaler.fit_transform(values)
#
## split into train and test sets
#n_train = int(0.6 * len(values))
#
#train = values[:n_train, :]
#test = values[n_train:, :]
#
## 7 feature at level, 8-15 feature at diff
## train with 7 feature  
#train_X, train_y = train[:, :8], train[:, -1]
#test_X, test_y = test[:, :8], test[:, -1]
#print(train_X.shape, len(train_X), train_y.shape)
#
## Reshape input to 3D [sample, timesteps, feature]
#train_X = train_X.reshape((train_X.shape[0], 1, 8))
#test_X = test_X.reshape((test_X.shape[0], 1, 8))
#
#
## =============================== Build Model ===============================
#
### design network
#model = Sequential()
#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
##model.add(Dense(50, activation="sigmoid"))
#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
#model.add(Dense(1))
#adam = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999)
#model.compile(loss='mse', optimizer=adam)
### fit network
#history = model.fit(train_X, train_y, epochs=100, batch_size=32,
#                    validation_data=(test_X, test_y), verbose=2, shuffle=False)
### plot history
#plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')
#plt.legend()
#plt.show()
#
### loop 1 step ahead forecast 
#
#
#
#### make a prediction (Regression)
###yhat = model.predict(test_X)
### for Classification ploblem
##yhat = model.predict_classes(test_X)
##test_X = test_X.reshape((test_X.shape[0], 8))
### invert scaling for forecast
##inv_yhat = np.concatenate((yhat, test_X[:, -8:]), axis=1)
##inv_yhat = scaler.inverse_transform(inv_yhat)
##inv_yhat = inv_yhat[:,0]
### invert scaling for actual
##test_y = test_y.reshape((len(test_y), 1))
##inv_y = np.concatenate((test_y, test_X[:, -8:]), axis=1)
##inv_y = scaler.inverse_transform(inv_y)
##inv_y = inv_y[:,0]
### calculate RMSE
##rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
##print('Test RMSE: %.3f' % rmse)