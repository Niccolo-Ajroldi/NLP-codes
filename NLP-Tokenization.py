# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 23:26:20 2022

@author: nicco
"""

import numpy as np
from keras.preprocessing.text import Tokenizer


samples = ['The cat sat on the mat.', 
           'The dog ate my homework.']

# create a tokenizer, that takes into account only the num_words most common words
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)

# named list mapping words to their rank/index (int)
tokenizer.word_index  
print('Found %s unique tokens.' % len(tokenizer.word_index))
# named list mapping words to the number of times they appeared on during fit
tokenizer.word_counts 
# number of documents (texts/sequences) the tokenizer was trained on
tokenizer.document_count

# Turns strings into lists of integer indices
sequences = tokenizer.texts_to_sequences(samples)
sequences

#------------------------------------------------------------------------------
# One Hot Encoding

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
one_hot_results

#------------------------------------------------------------------------------
# One Hot Hashing trick

# set *a-priori* the size of the vector used to represent sentences
# if you have close to 1,000 words (or more), you’ll see many hash collisions,
# which will decrease the accuracy of this encoding method
dimensionality = 1000

# maximul length of the sentence
max_length = 10

results = np.zeros((len(samples), max_length, dimensionality))

# iterate over sentences
for i, sample in enumerate(samples):
    # iterate over words in the sentence:
    for j, word in list(enumerate(sample.split()))[:max_length]:
        # hash the word to a random index from 0 to 1,000
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1.

results.shape
results[0].shape
results[1].shape

a = results[0]

#------------------------------------------------------------------------------
# Check

for i, sample in enumerate(samples):
    print(i, sample)
# iterate over sentences
# i = 0, sample = The cat sat on the mat.
# i = 1, sample = The cat sat on the mat.   

for j, word in list(enumerate(sample.split()))[:max_length]:
    print(j, word)
# j word
# 0 The
# 1 dog
# 2 ate
# 3 my
# 4 homework




