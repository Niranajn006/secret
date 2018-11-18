# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 10:36:06 2018

@author: User
"""
import pandas as pd
import numpy as np
from keras.layers import Input, Dense, LSTM
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers #Improve wight update avoid overfit
import pywt
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
def wavelet_processing(df, year_train, last_col):
    train_data = []
    train_data_range = len(df[df.index.year <= year_train])
    test_data_range = len(df[df.index.year == year_train + 1])
#  This code below for wavelet transform
    for i in range(len(df)):
        train = []
        for j in range(0, last_col):
            x = np.array(df.iloc[i, j])
            train = np.append(train, x)
        train_data.append(train)
    trained = pd.DataFrame(train_data, index=None)
    pre_en_train = pd.DataFrame(trained[0:train_data_range], index=None)
    pre_en_test = pd.DataFrame(trained[train_data_range+1:train_data_range+test_data_range-1], index=None)
    label_train = pd.DataFrame(np.array(df.iloc[0:train_data_range, -1]), index=None)
    label_test = pd.DataFrame(np.array(df.iloc[train_data_range+1:train_data_range+test_data_range-1, -1]), index=None)
    return pre_en_train, pre_en_test, label_train, label_test
    
#TODO: 3) Pre-Processing Data (Autoencoder) self supervise
def autoencoder_processing(data, ratio, encoded_dim, input_shape, 
                           encoded1_shape, encoded2_shape, decoded1_shape, decoded2_shape):
    
    # Construct Stackautoencoder
    input_data = Input(shape=(1, input_shape))
    encoded1 = Dense(encoded1_shape, activation="relu",activity_regularizer=regularizers.l2(10e-5))(input_data)
    encoded2 = Dense(encoded2_shape, activation="relu",activity_regularizer=regularizers.l2(10e-5))(encoded1)
    encoded3 = Dense(encoded_dim, activation="relu",activity_regularizer=regularizers.l2(10e-5))(encoded2)
    decoded1 = Dense(decoded1_shape, activation="relu",activity_regularizer=regularizers.l2(10e-5))(encoded3)
    decoded2 = Dense(decoded2_shape, activation="relu",activity_regularizer=regularizers.l2(10e-5))(decoded1)
    decoded = Dense(input_shape, activation="sigmoid")(decoded2)
    autoencoder = Model(inputs=input_data, outputs=decoded) # Evaluate  
    encoder = Model(input_data, encoded3) # use this model compress data to LSTM
    
    # Manage Data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_raw = scaler.fit_transform(data.values)
    scaled_data = np.reshape(scaled_raw, (len(data), 1, input_shape))
    train_data = scaled_data[0:int(len(data)*ratio)]
    test_data = scaled_data[int(len(data)*ratio)+1:]
    
    # Train model
    #cutom_adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.9, decay=0.0) 
    autoencoder.compile(loss='mean_absolute_error', optimizer='adam')
    history = autoencoder.fit(train_data, train_data, epochs=100,
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
    
    return pd.DataFrame(coded_train)
    
#TODO: 3) Train LSTM Model
def train_lstm_model(input_shape, train_data, train_label_data, test_data, test_label_data):
    
    # Construct LSTM
    input_data = Input(shape=(1, input_shape))
    lstm = LSTM(5, input_shape=(1, input_shape), return_sequences=True, activity_regularizer=regularizers.l2(0.003),
               recurrent_regularizer=regularizers.l2(0), dropout=0.2, recurrent_dropout=0.2)(input_data)
    perc = Dense(5, activation="sigmoid", activity_regularizer=regularizers.l2(0.005))(lstm)
    lstm2 =LSTM(2, activity_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.001),
                dropout=0.2, recurrent_dropout=0.2)(perc)
    output = Dense(1, activation="sigmoid", activity_regularizer=regularizers.l2(0.001))(lstm2)
    model = Model(input_data, output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    
    # Manage Data to LSTM input [sample, step, feature]
    train_x = np.reshape(np.array(train_data), (train_data.shape[0], 1, train_data.shape[1]))
    train_y = np.array(train_label_data)
    test_x = np.reshape(np.array(test_data), (test_data.shape[0], 1, test_data.shape[1]))
    test_y = np.array(test_label_data)
    
    # Train model
    model.fit(train_x, train_y, epochs=100)
    
    # Predict 
    prediction_data = []
    
    for i in range(len(test_y)):
        prediction = (model.predict(np.reshape(test_x[i], (1, 1, input_shape))))
        prediction_data.append(np.reshape(prediction, (1,)))
        prediction_corrected = (prediction_data - np.mean(prediction_data)) / np.std(prediction_data)
    
    plt.plot(prediction_corrected)
    
    return model
    
#TODO: 4) Predict Value

    
if __name__ == '__main__':
    #raw_df = get_data(r'D:\Projects\Data\Mkt timing.xlsx', 'Level_lag', 'A:M')
    pre_en_train, pre_en_test, label_train, label_test = wavelet_processing(raw_df, 2010, 11)
    # Generate Autoencode X
    encoded_train = autoencoder_processing(pre_en_train, ratio=0.8, encoded_dim=5,
                                         input_shape=11, encoded1_shape=11,
                                         encoded2_shape=7,
                                         decoded1_shape=7,
                                         decoded2_shape=11)
    
    encoded_test = autoencoder_processing(pre_en_test, ratio=0.8, encoded_dim=5,
                                     input_shape=11, encoded1_shape=11,
                                     encoded2_shape=7,
                                     decoded1_shape=7,
                                     decoded2_shape=11)
    
    # Train LSTM model 
    lstm_test = train_lstm_model(5, encoded_train, label_train, encoded_test, label_test)
    
    #df_train = raw_df[raw_df.index.year <= 2010]
    #df_test = raw_df[raw_df.index.year == 2011]