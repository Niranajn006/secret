# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from keras.layers import Input, Dense, LSTM
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import regularizers #Improve wight update avoid overfit
import pywt
from talib import MACD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use('ggplot')


#TODO: 1) Get Data 
def get_data(path, sheet, column):
    df = pd.read_excel(path, sheet_name=sheet, index_col='Dates',
                       usecols=column)
    return df

#TODO: 2) Pre-Processing Data (WaveletTransform)
def wavelet_processing(df, year_train, last_col, lookback):
    
    train_data = []
    train_data_range = len(df[df.index.year <= year_train])
    test_data_range = len(df[df.index.year == year_train + 1])
    
#  This code below for wavelet transform
    for i in range(len(df)):
        train = []
        for j in range(0, last_col):
            x = np.array(df.iloc[i:i+lookback, j])
#            (ca, cd) = pywt.dwt(x, "haar", 'smooth')
#            #cat = pywt.threshold(ca, np.std(ca), mode="soft")
#            #cdt = pywt.threshold(cd, np.std(cd), mode="soft")
#            tx = pywt.idwt(ca, cd, "haar")
            diff_tx = np.diff(np.log(x))
            train = np.append(train, diff_tx)
        train_data.append(train)
    trained = pd.DataFrame(train_data, index=None)

    pre_en_train = pd.DataFrame(trained[0:train_data_range], index=None)
#    pre_en_train = pre_en_train.shift(-1)
    pre_en_train = pre_en_train.fillna(0)
    pre_en_test = pd.DataFrame(trained[train_data_range+1:train_data_range+test_data_range], index=None)
    pre_en_test = pre_en_test.fillna(0)
    
#    for i in range(len(df)):
#        y = df.iloc[i+lookback+1, 1] / df.iloc[i+lookback, 1]
#        label.append(y)
#    log_re = np.where(np.diff(np.log(df.iloc[:,0])) > 0, 1, -1)
    log_re = np.diff(np.log(df.iloc[:, 0]))
    label_train = pd.DataFrame(log_re[0:train_data_range], index=None)
    label_test = pd.DataFrame(log_re[train_data_range+1:train_data_range+test_data_range], index=None)
    return pre_en_train, pre_en_test, label_train, label_test
    
#TODO: 3) Pre-Processing Data (Autoencoder) self supervise
def autoencoder_processing(data, ratio, encoded_dim, input_shape, 
                           encoded1_shape, encoded2_shape, decoded1_shape, decoded2_shape):
    
    # Construct Stackautoencoder
    input_data = Input(shape=(1, input_shape))
    encoded1 = Dense(encoded1_shape, activation="relu",activity_regularizer=regularizers.l2(10e-5))(input_data)
    encoded2 = Dense(encoded2_shape, activation="relu",activity_regularizer=regularizers.l2(10e-5))(encoded1)
    encoded3 = Dense(encoded_dim, activation="tanh",activity_regularizer=regularizers.l2(10e-5))(encoded2)
    decoded1 = Dense(decoded1_shape, activation="relu",activity_regularizer=regularizers.l2(10e-5))(encoded3)
    decoded2 = Dense(decoded2_shape, activation="relu",activity_regularizer=regularizers.l2(10e-5))(decoded1)
    decoded = Dense(input_shape, activation="tanh")(decoded2)
    autoencoder = Model(inputs=input_data, outputs=decoded) # Evaluate  
    encoder = Model(input_data, encoded3) # use this model compress data to LSTM
    
    # Manage Data
    scaled_raw = data.values
    scaled_data = np.reshape(scaled_raw, (len(data), 1, input_shape))
    train_data = scaled_data[0:int(len(data)*ratio)]
    test_data = scaled_data[int(len(data)*ratio)+1:]
    
    # Train model
    #cutom_adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.9, decay=0.0) 
    autoencoder.compile(loss='mean_absolute_error', optimizer='adam')
    autoencoder.fit(train_data, train_data, epochs=100,
                    validation_data=(test_data, test_data))
#    print(autoencoder.evaluate(test_data, test_data))
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
    # Use encode generate encode data before LSTM
    coded_train = []
    for i in range(len(data)):
        x = scaled_raw[i, :]
        values = np.reshape(x, (1, 1, input_shape))
        coded = encoder.predict(values)
        shaped = np.reshape(coded, (encoded_dim,))
        coded_train.append(shaped)
    
    df_coded = pd.DataFrame(coded_train)
#    coded_train = scaler.inverse_transform(np.array(coded_train))
    
    return df_coded
    
#TODO: 3) Train LSTM Model
def train_lstm_model(input_shape, train_data, train_label_data,
                     test_data, test_label_data, predict_level=False):
    
    # Construct LSTM with Functional API
    input_data = Input(shape=(1, input_shape))
    lstm = LSTM(5, input_shape=(1, input_shape), return_sequences=True, activity_regularizer=regularizers.l2(0.003),
                   recurrent_regularizer=regularizers.l2(0), dropout=0.2)(input_data)
    hidden1 = Dense(5,activation="tanh", activity_regularizer=regularizers.l2(0.005))(lstm)
    lstm2 = LSTM(2, activity_regularizer=regularizers.l2(0.01),
                    dropout=0.2, recurrent_dropout=0.2)(hidden1)
#    hidden2 = Dense(2, activity_regularizer=regularizers.l2(0.001))(lstm2)
    output = Dense(1, activation="tanh", activity_regularizer=regularizers.l2(0.001))(lstm2)
    model = Model(input_data, output)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    # Construct LSTM with Sequential Constructor
#    model = Sequential()
#    model.add(LSTM(5, batch_input_shape=(1, 1, input_shape)))
#    model.add(Dense(1))
#    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    
    # Manage Data to LSTM input [sample, step, feature]
    train_x = np.reshape(np.array(train_data), (train_data.shape[0], 1, train_data.shape[1]))
    train_y = np.array(train_label_data)
    test_x = np.reshape(np.array(test_data), (test_data.shape[0], 1, test_data.shape[1]))
    test_y = np.array(test_label_data)
    
    # Train model
    history = model.fit(train_x, train_y, epochs=10, shuffle=False)
    
    plt.plot(history.history['loss'])
    print('Model Performance: ',model.evaluate(test_x, test_y))
    
    # Predict 
    prediction_data = []
    prediction_level = []
    
    for i in range(len(test_y)):
        # Predict regression
        prediction = (model.predict(np.reshape(test_x[i], (1, 1, input_shape))))
        prediction_data.append(np.reshape(prediction, (1,)))
        #level = np.exp(np.reshape(prediction, (1,))) * 
#    plt.plot(prediction_corrected)
    
    plt.figure()
    plt.plot(prediction_data)
    #predict_class = np.where(np.array(prediction_data) > 0.5, 1, 0) 
    
    return model, prediction_data
    
#TODO: 4) Alternative Model (Multivariate Seq-to-Seq Encode Decode)

    
if __name__ == '__main__':
    raw_df = get_data(r'D:\Project files\Data\Mkt timing.xlsx', 'Level_lag', 'A:K')
    
        
#    macd, macdsignal, macdhist = MACD(raw_df['ACWI'],fastperiod=12, slowperiod=26, signalperiod=9)
#    
#    macd = pd.Series(macdhist, name='MACD')
#    raw_df = raw_df.join(macd)
#    df = raw_df.dropna(inplace=True)
    
    #for rolling sample 
    #raw_df[(raw_df.index.year >=2010) & (raw_df.index.year <=2012)]
    
    # Denoise data and manipulate data
    pre_en_train, pre_en_test, label_train, label_test = wavelet_processing(raw_df, 2010, 10 , 2)
    
#    print('-'*10, "Autoencoder Process", '-'*10)
#    
#    #Generate Autoencode X
#    encoded_train = autoencoder_processing(pre_en_train, ratio=0.8, encoded_dim=10,
#                                         input_shape=50, encoded1_shape=30,
#                                         encoded2_shape=20,
#                                         decoded1_shape=20,
#                                         decoded2_shape=30)
#    
#    encoded_test = autoencoder_processing(pre_en_test, ratio=0.8, encoded_dim=10,
#                                     input_shape=50, encoded1_shape=30,
#                                     encoded2_shape=20,
#                                     decoded1_shape=20,
#                                     decoded2_shape=30)
    
    print('-'*10, "LSTM Process", '-'*10)
    
    # Train LSTM model 
    lstm_test, prediction = train_lstm_model(10, pre_en_train, label_train, pre_en_test, label_test)
    
#    plt.subplot(211)
    plt.plot(pd.DataFrame(prediction), label='LSTM Predict')
#    plt.gca().set_title('Predict')
#    plt.subplot(212)
    plt.plot(label_test, label='Diff log ACWI')
#    plt.gca().set_title('ACWI diff log')
    
    #df_train = raw_df[raw_df.index.year <= 2010]
    #df_test = raw_df[raw_df.index.year == 2011]