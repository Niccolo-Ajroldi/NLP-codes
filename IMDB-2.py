# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 19:16:01 2022

@author: nicco
"""

from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000
maxlen = 500
batch_size = 32

(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)

print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')
print('Pad sequences (samples x time)')

input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)

print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

#%% TRAIN

from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

model = Sequential()

# Turns positive integers (indexes) into dense vectors of fixed size
# This layer can only be used as the first layer in a model
model.add(Embedding(10000, 32))

# SimpleRNN layer
model.add(SimpleRNN(32))
model.summary()

# stack multiple RNN layers
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))
model.summary()






