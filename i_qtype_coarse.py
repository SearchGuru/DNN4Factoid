from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing.text import Tokenizer
from six.moves import cPickle
from keras.models import model_from_json

import deepctxt_util
from deepctxt_util import DCTokenizer

maxlen = 25  # cut texts after this number of words (among top max_features most common words)

tokenizer = DCTokenizer()
print('Loading tokenizer')
tokenizer.load('./glove.6B.100d.txt')
#tokenizer.load('./glove.42B.300d.txt')
print('Done')

print('Loading model')
with open("./coarse_type_model_lstm_glove_100b.json", "r") as f:
    json_string = f.readline()
    model = model_from_json(json_string)
print('Done')

print('Compile model')
model.compile(loss='categorical_crossentropy', optimizer='adam')
print('Done')

print('Loading weights')
model.load_weights('./coarse_type_model_lstm_glove_100b.h5')
print('Done')

idx2type = {0:"DESCRIPTION", 1:"NUMERIC", 2:"ENTITY", 3:"PERSON", 4:"LOCATION"}

while True:
    print("===============================================")
    query = raw_input('Enter query: ')
    X1 = []
    X1.append(query)
    X2 = tokenizer.texts_to_sequences(X1, maxlen)
    X = sequence.pad_sequences(X2, maxlen=maxlen)
    pred = model.predict_proba(X, batch_size=1)
    idx = np.argmax(pred[0])
    print("Type=" + idx2type[idx])
    print(pred)
