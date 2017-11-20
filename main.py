# Load Larger LSTM network and generate text
import sys
import re
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import stop_words

# load ascii text and covert to lowercase
filename = "DF/cleaned_posts.txt"
raw_text = open(filename).read()
stoplist = stop_words.get_stop_words('danish')

raw_text = " ".join([word for word in raw_text.split() if word not in stoplist])
raw_text = raw_text.lower()
raw_text = re.sub('\n', '', raw_text)


# filename = "text_list_file.txt"
# raw_text = open(filename).read()
# raw_text = raw_text.lower()
# raw_text = re.sub('\n','',raw_text)
# raw_text = re.sub('[1234567890]?', '', raw_text)
# lines = [line.lower().rstrip('\n').rstrip('\t') for line in open(filename)]
# lines = [line for line in lines if len(line)>0]
# lines = [re.sub('[(){}<>]!,:;-<>/&=|', '', post) for post in lines]
# lines = [re.sub('[1234567890]?', '', post) for post in lines]

def addspacestosentence(instring,seqmax):
	while len(instring)<seqmax:
		instring+=" "
	return instring
#
# filename = "wonderland.txt"
# raw_text = open(filename).read()
# raw_text = raw_text.lower()

# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print "Total Characters: ", n_chars
print "Total Vocab: ", n_vocab
# prepare the dataset of input to output pairs encoded as integers - ask changed from 100
# seq_length = 20
# seq_max = 60
# dataX = []
# dataY = []
# ## This was how it was done originally:
# for i in range(len(lines)):
# 	for ichar in range(0, min(len(lines[i]), seq_max), 1):
# 		if (seq_length + ichar) >= len(lines[i]): break
# 		seq_in = lines[i][ichar:ichar + seq_length]  # chars in sentence up 'till last
# 		seq_out = lines[i][ichar + seq_length]  # last char in sentence
# 		seq_in = addspacestosentence(seq_in, seq_max)
# 		dataX.append([char_to_int[ic] for ic in seq_in])
# 		dataY.append(char_to_int[seq_out])
#
# for i in range(len(dataX)):
# 	if (len(dataX[i]) == 0):
# 		dataX.pop(i)
# 		dataY.pop(i)
# n_patterns = len(dataX)
# print "Total Patterns: ", n_patterns

# prepare the dataset of input to output pairs encoded as integers - ask changed from 100
seq_length = 50
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print "Total Patterns: ", n_patterns
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

#LSTM: Long short-term memory - recurrent neural network https://en.wikipedia.org/wiki/Long_short-term_memory
# define the LSTM model (RNNS allows forward and backward connections between neurons, eg. learn from which words or letters precede and follow a given word/letter.
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
# load the network weights
#filename = "weights-improvement-47-1.2219-bigger.hdf5"
#filename = "weights-improvement-03-2.6503.hdf5"
#filename="fewerchars/weights-improvement-37-1.4256-bigger.hdf5"
#filename="fewerchars/weights-improvement-63-0.1519-bigger.hdf5"
#filename="DF/weights-improvement-05-1.9859-bigger.hdf5"
filename="DF/weights-improvement-04-2.1937-bigger.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# generate characters
for itry in range(0,5):
	print "triel " + str(itry) + "\n"

	# pick a random seed
	start = numpy.random.randint(0, len(dataX) - 1)
	pattern = dataX[start]
	print "Seed:"
	# pattern = [char_to_int[ichar] for ichar in 'lutningen af en sekvens fremtiden for danskhed er ']
	print "\"", ''.join([int_to_char[value] for value in pattern]), "\""
	for i in range(100):
		x = numpy.reshape(pattern, (1, len(pattern), 1))
		x = x / float(n_vocab)
		prediction = model.predict(x, verbose=0)
		index = numpy.argmax(prediction)
		result = int_to_char[index]
		seq_in = [int_to_char[value] for value in pattern]
		sys.stdout.write(result)
		pattern.append(index)
		pattern = pattern[1:len(pattern)]
print "\nDone."