from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from collections import Counter
# load ascii text and covert to lowercase

lille_alfabet = 'abcdefghijklmnopqrstuvwxyz '

def clean_post(post):
    post=post.replace('\n',' ').replace('\r',' ')
    post=post.lower()
    post = ''.join(c for c in post if c in lille_alfabet)
    return post


filename = "./roed/cleaned_posts.txt"
raw_text = open(filename).read()
raw_text =  clean_post(raw_text).split(' ')
word_counter = Counter(raw_text)
# create mapping of unique chars to integers, and a reverse mapping
words = sorted(list(set(raw_text)))
word_to_int = dict((c, i) for i, c in enumerate(words))
# summarize the loaded data
n_words = len(raw_text)
n_vocab = len(words)
print "Total Characters: ", n_words
print "Total Vocab: ", n_vocab
# prepare the dataset of input to output pairs encoded as integers
seq_length = 5
dataX = []
dataY = []
for i in range(0, n_words - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    if word_counter[seq_out] > 10:
        dataX.append([word_to_int[word] for word in seq_in])
        dataY.append(word_to_int[seq_out])
n_patterns = len(dataX)
print "Total Patterns: ", n_patterns
# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
#y = np_utils.to_categorical(dataY)
y = np.array(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[0], activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="roed/weights-improvement_SUC-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, nb_epoch=1000, batch_size=64,callbacks=callbacks_list) #sparse_categorical_crossentropy

# Try batch by batsh
# Try sparse_categorical_crossentropy






