# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 19:16:01 2022

@author: nicco
"""

from keras.datasets import imdb
from keras.preprocessing import sequence

#%% DATA

max_features = 10000
maxlen = 500
batch_size = 32

(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)

print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

# each element of the np.array is a list, containing indexes of the words in that sentence
type(input_train[0])
input_train[0][:10]

# Different sequences have differences length (different number of words)
len(input_train[0])
len(input_train[1])

# Pad sequences (samples x maxlen) -> Fill with zeros, or truncate
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test  = sequence.pad_sequences(input_test,  maxlen=maxlen)

print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

#%% RNN

from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

from keras.layers import Dense

#%% RNN 1 layer

model = Sequential()

# Turns positive integers (indexes) into dense vectors of fixed size
# This layer can only be used as the first layer in a model
model.add(Embedding(input_dim=max_features, output_dim=32))

# SimpleRNN layer
model.add(SimpleRNN(units=10, return_sequences=False))

# Add dense layer, with 1 output for classification
model.add(Dense(1, activation='sigmoid'))

# compile and fit
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#%% RNN 1 layer, experiment

model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=32))
model.add(SimpleRNN(units=10, return_sequences=True))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

# The model declaration is inconsistent: cannot put a Dense layer of size 1
# after a RNN layer with return_sequence=True
# The subsequent Dense layer does only have one node, and hence will
# only use the output of the last time of the RNN.

# If I wanted to feed all the outputs of the previous RNN layer to the Dense one I should use:
model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=32))
model.add(SimpleRNN(units=10, return_sequences=True))
model.add(Dense(10, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
    

#%% RNN - Multiple Layers

model = Sequential()

# Turns positive integers (indexes) into dense vectors of fixed size
# This layer can only be used as the first layer in a model
model.add(Embedding(input_dim=max_features, output_dim=32))

# SimpleRNN layer
model.add(SimpleRNN(units=10, return_sequences=True))

# SimpleRNN layer
model.add(SimpleRNN(units=15, return_sequences=True))

# ValueError !!!
# Cannot stack an RNN upon a layer with a "single output"

#%% RNN - Multiple Layers

model = Sequential()

# Turns positive integers (indexes) into dense vectors of fixed size
# This layer can only be used as the first layer in a model
model.add(Embedding(input_dim=max_features, output_dim=32))

# SimpleRNN layer
model.add(SimpleRNN(units=10, return_sequences=False))

# SimpleRNN layer
model.add(SimpleRNN(units=15, return_sequences=True))

# ValueError !!!
# Cannot stack an RNN upon a layer with a "single output"

#%% LSTM

from keras.layers import LSTM

model = Sequential()
model.add(Embedding(max_features, 32))

# 32 is the output dimensionality of the LSTM layer
model.add(LSTM(32))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['acc'])

history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)














