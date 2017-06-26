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
from sklearn.model_selection import KFold
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

Xs,Ys = read_data("/Users/macbook/Downloads/tmp.csv")
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

predicted_list = []
true_list = []


numpy.random.shuffle(a)

kf = KFold(n_splits=10)

for train_idx, test_idx in kf.split(a):
    X_train = numpy.array([a[i] for i in train_idx])
    Y_train = numpy.array([Ys[i] for i in train_idx])

    X_test = numpy.array([a[i] for i in test_idx])
    Y_test = numpy.array([Ys[i] for i in test_idx])

    # create the model
    model = Sequential()
    model.add(Bidirectional (LSTM(100,recurrent_dropout=0.1), input_shape=(3000/n,n)))
    #model.add(Bidirectional(LSTM(100, return_sequences=True)))
    #model.add(Dropout(0.3))
    #model.add(Bidirectional(LSTM(25)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=20)
    predictions = model.predict(X_test)
    rounded = []


    for i in range(0, len(Y_test)):
        if predictions[i] > 0.5:
            rounded.append(1)
            predicted_list.append(1)
        else:
            rounded.append(0)
            predicted_list.append(0)
        true_list.append(Y_test[i])


    recall_0_list.append(recall_score(Y_test, rounded, pos_label=0))
    recall_1_list.append(recall_score(Y_test, rounded, pos_label=1))
#    auc.append(roc_auc_score(Y_test, rounded))

print "RESULTS:"
print "RECALL 0:", sum(recall_0_list) / float(len(recall_0_list))
print "RECALL 1:", sum(recall_1_list) / float(len(recall_1_list))


print "RESULTS flattened:"
print "RECALL 0:", recall_score(true_list, predicted_list, pos_label=0)
print "RECALL 1:", recall_score(true_list, predicted_list, pos_label=1)
print "AUC :", roc_auc_score(true_list, predicted_list)





