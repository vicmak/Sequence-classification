import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Masking
from keras.layers import Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from collections import Counter, defaultdict
from itertools import count
import nltk
import mmap

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from keras.layers import Dropout
from keras.layers import Bidirectional

n=50

def read_data(filename):
    Xs = []
    Ys = []
    with open(filename) as f:
        for line in f:
            tokens = line.split(",")
            x = tokens[1:]
            x = [float(i) for i in x]
            missing_values_number = 3000 - len(x)
            for i in range(0,missing_values_number):
                x.append(0.1)
            chunked_x = [x[i:i+n] for i in xrange(0, len(x),n)]
            y = int(tokens[0])
            Xs.append(chunked_x)
            Ys.append(y)
    return Xs,Ys

Xs = [] #List of 56 lists of 60 lists of 50 numbers
Ys = [] #classification 1 or 0

Xs,Ys = read_data("/Users/macbook/Downloads/tmp.txt")
print Xs[0]
Xs = numpy.array(Xs)
print "SHAPE", Xs.shape

instances_number = len(Xs)
sequence_length = len(Xs[0])
features_number = len(Xs[0][0])

a = numpy.empty((instances_number,sequence_length,features_number))

for i in range(0,instances_number):
    for j in range(0, sequence_length):
        for k in range(0, features_number):
            a[i][j][k] = Xs[i][j][k]

data_train, data_test, labels_train, labels_test = train_test_split(a, Ys, test_size=0.10, random_state=42)

recall_1_list = []
recall_0_list = []
auc = []

# create the model
model = Sequential()
model.add(LSTM(50, input_shape=(60,50)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(data_train, labels_train, epochs=10)

predictions = model.predict(data_test)

auc.append(roc_auc_score(labels_test, predictions))

rounded = []
for pred in predictions:
    if pred > 0.5:
        rounded.append(1)
    else:
        rounded.append(0)

recall_0_list.append(recall_score(labels_test, rounded, pos_label=0))
recall_1_list.append(recall_score(labels_test, rounded, pos_label=1))


print "RESULTS:"
print "RECALL 0:", sum(recall_0_list) / float(len(recall_0_list))
print "RECALL 1:", sum(recall_1_list) / float(len(recall_1_list))
print "AUC :", sum(auc)/float(len(auc))


