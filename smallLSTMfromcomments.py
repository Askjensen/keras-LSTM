# Small LSTM Network to Generate Text for Alice in Wonderland
import numpy
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
# load ascii text and covert to lowercase
filename = "text_list_file.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
raw_text = re.sub('\n','',raw_text)
raw_text = re.sub('[1234567890]?', '', raw_text)
lines = [line.lower().rstrip('\n').rstrip('\t') for line in open(filename)]
lines = [line for line in lines if len(line)>0]
lines = [re.sub('[(){}<>]!,:;-<>/&=|', '', post) for post in lines]
lines = [re.sub('[1234567890]?', '', post) for post in lines]

def addspacestosentence(instring,seqmax):
	while len(instring)<seqmax:
		instring+=" "
	return instring

# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print "Total Characters: ", n_chars
print "Total Vocab: ", n_vocab
# prepare the dataset of input to output pairs encoded as integers
seq_length = 20
seq_max = 60
dataX = []
dataY = []
## This was how it was done originally:
for i in range(len(lines)):
	for ichar in range(0,min(len(lines[i]),seq_max),1):
		if (seq_length+ichar) >= len(lines[i]): break
		seq_in = lines[i][ichar:ichar+seq_length] #chars in sentence up 'till last
		seq_out = lines[i][ichar+seq_length] #last char in sentence
		seq_in = addspacestosentence(seq_in,seq_max)
		dataX.append([char_to_int[ic] for ic in seq_in])
		dataY.append(char_to_int[seq_out])

for i in range(len(dataX)):
	if(len(dataX[i])==0):
		dataX.pop(i)
		dataY.pop(i)
n_patterns = len(dataX)
print "Total Patterns: ", n_patterns
# reshape X to be [samples, time steps, features]
#ask changed sequence length to
X = numpy.reshape(dataX, (n_patterns, seq_max, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="commentssqe30/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, nb_epoch=100, batch_size=128, callbacks=callbacks_list)