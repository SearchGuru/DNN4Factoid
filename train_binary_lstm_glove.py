from __future__ import print_function
import numpy as np
np.random.seed(1337)  

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from six.moves import cPickle
import os
import string

import deepctxt_util
from deepctxt_util import DCTokenizer

maxlen = 25 # cut texts after this number of words (among top max_features most common words)
batch_size = 100
epoch = 3

tokenizer = DCTokenizer()
print('Loading tokenizer')
tokenizer.load('./glove.6B.100d.txt')
#tokenizer.load('./glove.42B.300d.txt')
print('Done')

max_features = tokenizer.n_symbols
vocab_dim = tokenizer.vocab_dim

print('Loading data... (Train)')
(X1, y_train) = deepctxt_util.load_raw_data_x_y(path='./raw_data/person_birthday_deep_learning_training_rawquery_cleaned.2.tsv', y_shift = 0)
print('Done')

print('Loading data... (Validation)')
(X3, y_val) = deepctxt_util.load_raw_data_x_y(path='./raw_data/person_birthday_deep_learning_val_rawquery_cleaned.tsv', y_shift = 0)
print('Done')

print('Loading data... (Test)')
(X2, y_test) = deepctxt_util.load_raw_data_x_y(path='./raw_data/person_birthday_deep_learning_eval_rawquery_cleaned.tsv', y_shift = 0)
print('Done')

print('Converting data... (Train)')
X_train = tokenizer.texts_to_sequences(X1, maxlen)
print('Done')

print('Converting data... (Validation)')
X_val = tokenizer.texts_to_sequences(X3, maxlen)
print('Done')

print('Converting data... (Test)')
X_test = tokenizer.texts_to_sequences(X2, maxlen)
print('Done')

print(len(X_train), 'y_train sequences')
print(len(X_val), 'y_val sequences')
print(len(X_test), 'y_test sequences')

nb_classes = np.max(y_train)+1
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_val = np_utils.to_categorical(y_val, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print('Y_train shape:', Y_train.shape)
print('Y_val shape:', Y_val.shape)
print('Y_test shape:', Y_test.shape)

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_val = sequence.pad_sequences(X_val, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_val shape:', X_val.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=vocab_dim, input_length=maxlen, weights=[tokenizer.embedding_weights]))
model.add(LSTM(128))  # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam')

print("Train...")
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1,
           validation_data=(X_val, Y_val))


print("Evluating the model...\n")
acc = model.evaluate(X_test, Y_test, batch_size=100)
print('Test accuracy:', acc)

json_model_string = model.to_json()
with open("./person_birthday_binary_model_lstm_glove_"+str(batch_size)+"b.json", "w") as f:
    f.write(json_model_string)
model.save_weights("./person_birthday_binary_model_lstm_glove_" + str(batch_size) + "b.h5")


