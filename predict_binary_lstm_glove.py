from __future__ import print_function
import numpy as np
np.random.seed(1337)  

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from six.moves import cPickle

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

print('Loading data... (Test)')
#(X2, y_test) = deepctxt_util.load_raw_data_x_y(path='./raw_data/bing_query.tsv', y_shift=0)
(X2, y_test) = deepctxt_util.load_raw_data_x_y(path='./raw_data/person_birthday_deep_learning_eval_rawquery_cleaned.tsv', y_shift=0)
print('Done')


print('Converting data... (Test)')
X_test = tokenizer.texts_to_sequences(X2, maxlen)
print('Done')

print(len(X_test), 'y_test sequences')

nb_classes = np.max(y_test)+1
Y_test = np_utils.to_categorical(y_test, nb_classes)

print('Y_test shape:', Y_test.shape)

print("Pad sequences (samples x time)")
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_test shape:', X_test.shape)

print('Load model...')

file = open('./person_birthday_binary_model_lstm_glove_100b.json', 'rb')
model_string = file.read()
file.close()
model = model_from_json(model_string)
model.load_weights('./person_birthday_binary_model_lstm_glove_100b.h5')


print('Predict LSTM model...')
#Y_pred = model.predict_classes(X_test, batch_size=100, verbose=1)
Y_prob = model.predict_proba(X_test, batch_size=100, verbose=1)

out_file = open('predict_out_bing_prob_general.tsv', 'wb')
for i in range(0, len(Y_prob)):
    s = X2[i] + '\t' + str(y_test[i]) + '\t' + str(Y_prob[i][0]) + '\t' + str(Y_prob[i][1]) + '\n'
    out_file.write(s)
out_file.close()

