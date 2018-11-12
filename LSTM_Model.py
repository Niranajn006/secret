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



# =============================== Initail Model ===============================
df = pd.read_excel(r'D:\Project files\FXProjects\Mkt_timing.xlsx',
                   sheet_name='Chg_lag',
                   index_col='Dates',
                   usecols="A:K")


values = df.values

# Convert values to float 
values = values.astype('float64')

# Encoder integer label to direction
encoder = LabelEncoder()
values[:, 9] = encoder.fit_transform(values[:, 9])

#values[:, 16] = to_categorical(values[:, 16])

# Normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(values)

# split into train and test sets
n_train = int(0.6 * len(values))

train = values[:n_train, :]
test = values[n_train:, :]

# 7 feature at level, 8-15 feature at diff
# train with 7 feature  
train_X, train_y = train[:, :9], train[:, -1]
test_X, test_y = test[:, :9], test[:, -1]
print(train_X.shape, len(train_X), train_y.shape)

# Reshape input to 3D [sample, timesteps, feature]
train_X = train_X.reshape((train_X.shape[0], 1, 9))
test_X = test_X.reshape((test_X.shape[0], 1, 9))


# =============================== Build Model ===============================

## design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
#model.add(Dense(50, activation="sigmoid"))
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
adam = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999)
model.compile(loss='mse', optimizer=adam)
## fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=32,
                    validation_data=(test_X, test_y), verbose=2, shuffle=False)
## plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

## make a prediction (Regression)
#yhat = model.predict(test_X)
# for Classification ploblem
yhat = model.predict_classes(test_X)
test_X = test_X.reshape((test_X.shape[0], 9))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, -9:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, -9:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)