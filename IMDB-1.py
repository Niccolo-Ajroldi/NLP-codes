
# coding: utf-8

# ## Loading data

#%%

from keras.datasets import imdb
import numpy as np

#%% DATA

# reviews have already been processed, 
# each review has been turned into a sequence of integers,
# each integer stands for a specific word in a dictionary 
# x_i = integers corresponding to words present in the review i
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# The argument num_words=10000 means you’ll only keep the top 10,000 most frequently
# occurring words in the training data. Rare words will be discarded. This allows
# you to work with vector data of manageable size.

type(train_data)
type(train_labels)
train_data.shape
train_labels.shape
len(train_data[1])
len(train_data[2])

train_data[0][:10]
train_labels[0]

max([max(sequence) for sequence in train_data])


#%% One hot ecoding

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

x_train[0]
len(x_train[1])
len(x_train[2])

y_train = np.asarray(train_labels).astype('float32')
y_test  = np.asarray(test_labels). astype('float32')


#%% Building the network

from keras import models
from keras import layers

#%% 1st MODEL

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# compile
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# TRAINING
# Train the model for 20 epochs (20 iterations over all samples in the
# x_train and y_train tensors), in mini-batches of 512 samples.
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

history_dict = history.history
history_dict.keys()
history_dict['accuracy']
history_dict['loss']

# PLOT 
import matplotlib.pyplot as plt
n_epochs = 20

# train and valdiation loss
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(n_epochs) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# clear plot
plt.clf()

# train and validation accuracy
acc_values = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Overfitting!

# same model as before
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# fit for only 4 epochs, fit on the entire training set
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

# generate prediction probabilities
model.predict(x_test)

#%% 2nd MODEL: tanh

# same model as before
model = models.Sequential()
model.add(layers.Dense(16, activation='tanh', input_shape=(10000,)))
model.add(layers.Dense(16, activation='tanh'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

n_epochs = 20

# train and valdiation loss
plt.clf()
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, n_epochs + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# train and validation accuracy
plt.clf()
acc_values = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# fit for only 4 epochs, fit on the entre training set
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

#%% 2nd MODEL: tanh








































