# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 22:45:18 2018

@author: User
"""
#%% Import library
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import np_utils
# fix random seed for reproducibility
np.random.seed(7)

#%% Define dataset, We define the alphabet in uppercase characters for readability.
# Define the raw dataset 
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

seq_length = 1
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
    print(seq_in, '->', seq_out)
    
#%% reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (len(dataX), seq_length, 1))

# Normalize 
X = X / float(len(alphabet))    

# One hot encode the output variable
y = np_utils.to_categorical(dataY)

#%% (1) Naive LSTM for Learning One-Char to One-Char
# Each input-output pattern is shown to the network in a random order and the
# state of the network is reset after each pattern (each batch where each batch contain one pattern)

# 1 Layer LSTM with 32 units and output softmax activation function 
model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, batch_size=1, verbose=2)

# summarize performance of the odel
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" %(scores[1]*100))

# model prediction
for pattern in dataX:
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(seq_in , "->" , result)
    
#%% (2) Naive LSTM for a Three-Char 'Feature Window' to One-Char Mapping 
""" 
Popular approach to adding more context to data fro multilayer perceptrons is use
window method where previous steps in the sequence are provided as additional input features 
to the network
poor for this method 
"""
seq_length = 3 
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
    print(seq_in, '->', seq_out)

# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (len(dataX), 1, seq_length)) # <- Change this 

# Normalize 
X = X / float(len(alphabet))    

# One hot encode the output variable
y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, batch_size=1, verbose=2)

# summarize performance of the odel
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" %(scores[1]*100))

# model prediction
for pattern in dataX:
    x = np.reshape(pattern, (1, 1, len(pattern))) # <- Change this 
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(seq_in , "->" , result)
    
#%% (3) Naive LSTM for a Three-Char 'Time Step Window' to One-Char mapping
"""
 In Keras, the intended use of LSTM is to provide context in the form of time 
 steps, rather than windowd features like with other netwrok types
"""
seq_length = 3 
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
    print(seq_in, '->', seq_out)

# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (len(dataX), seq_length, 1)) # <- Change this 

# Normalize 
X = X / float(len(alphabet))    

# One hot encode the output variable
y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, batch_size=1, verbose=2)

# summarize performance of the odel
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" %(scores[1]*100))

# model prediction
for pattern in dataX:
    x = np.reshape(pattern, (1, len(pattern), 1)) # <- Change this 
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(seq_in , "->" , result)