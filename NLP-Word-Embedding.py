# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 19:32:48 2022
@author: nicco

"""

from keras.datasets import imdb
from keras import preprocessing

#%% DATA

# restrict the movie reviews to the top 10,000 most common words
max_features = 10000
maxlen = 20

# load data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

#%% LEARN EMBEDDING

from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

model = Sequential()

# Embedding layer: 
# 10,000 = number of possible tokens
# 8 = dimension of the embedding space
# outputs 3D floating point tensor of size (samples, maxlen, 8)
# EACH element (token/word) of each input sequence is mapped to a 8-dim vector!
# thus each input sequence, which is made by maxlen words/tokens is mapped to a (maxlen)x8 tensor
model.add(Embedding(10000, 8, input_length=maxlen))

# Flatten layer:
# squashes the 2D tensor input into a 2D tensor output
model.add(Flatten())

# Dense layer with sigmoid activation
model.add(Dense(1, activation='sigmoid'))

# compile model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

# FIT
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)


# The Flatten() layer is flattening the embedded sequence
# This means that, for each input made of maxlen words, the Flatten() layer
# maps it to a tensor of size 8x1
# Equivalently, a batch of size (samples, maxlen), is mapped to a 2D tensor
# of size (samples, 8x maxlen)
# 
# The subsequent Dense layer thus processes each embedded token/word independently!
# The next step would be to use a RNN or a CNN instead.


#%% PLOT EMBEDDING!




#------------------------------------------------------------------------------
# ACESS EMBEDDING WEIGHTS

#https://stackoverflow.com/questions/51235118/how-to-get-word-vectors-from-keras-embedding-layer

# or access the embedding layer through the constructed model 
# first `0` refers to the position of embedding layer in the `model`
embeddings = model.layers[0].get_weights()[0]
embeddings

# `embeddings` has a shape of (num_vocab, embedding_dim) 
embeddings.shape

# `word_index` is a mapping (i.e. dict) from words to their index, e.g. `love`: 69
word_index = imdb.get_word_index()
word_index

# create a dictionary that maps words embeddings to embedded vectors
words_embeddings = {}
for w, idx in word_index.items():
    if idx < embeddings.shape[0]:
        words_embeddings[w] = embeddings[idx]
words_embeddings

# now you can use it like this for example
print(words_embeddings['love'])  # possible output: [0.21, 0.56, ..., 0.65, 0.10]

# Check some words
words_embeddings['nice']
words_embeddings['beautiful']
words_embeddings['lovely']
words_embeddings['terrific']
words_embeddings['good']
words_embeddings['awful']
words_embeddings['bad']
words_embeddings['red']

import numpy as np

# check distance between words
np.linalg.norm(words_embeddings['nice'] - words_embeddings['beautiful'])
np.linalg.norm(words_embeddings['nice'] - words_embeddings['good'])
np.linalg.norm(words_embeddings['nice'] - words_embeddings['bad'])
np.linalg.norm(words_embeddings['good'] - words_embeddings['red'])


#------------------------------------------------------------------------------
# T-SNE

from sklearn.manifold import TSNE

X = embeddings
X.shape
X_embedded_2D = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(embeddings)
X_embedded_2D.shape


#------------------------------------------------------------------------------
# T-SNE PLOT
# TODO 

x_train.shape
x_train[0]

# decode the first review
word_index = imdb.get_word_index()

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]])

decoded_review

# I want two similar words
word_index['nice']
word_index['beautiful']
word_index['lovely']
word_index['terrific']
word_index['good']

word_index['awful']
word_index['bad']

word_index['red']




