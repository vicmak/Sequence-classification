import numpy
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from keras.layers import Bidirectional

import pickle
import pandas as pd

folder_path = '/Users/macbook/Desktop/corpora/VicDana'

'''
def load_obj(name):
    with open(folder_path + '/' + name + '.pkl', 'rb') as f:
        print f
        return pickle.load(f)


mts_signals = load_obj("mts_tslearn_format")

Xs = numpy.array(list(mts_signals['mts'])
              , dtype=numpy.dtype('float'))

df_windows_details = pd.read_csv(folder_path + 'df_windows_details.csv', index_col=0)
Ys = df_windows_details.label.values
'''

print "load"
df = pd.DataFrame(numpy.random.randint(0,100,size=(10, 6)))

print "DATAFRAME"
print df

def read_data():

    Xs = []
    Ys = []

    for i in range(0,1000):
        current_person_samples = []
        for j in range(0,60):
            current_sample = []
            for k in range(0,6):
                current_sample.append(random.random())
            current_person_samples.append(current_sample)
        Xs.append(current_person_samples)
        Ys.append(random.choice([0, 1]))

    return Xs,Ys

Xs = [] #List of 56 lists of 60 lists of 50 numbers
Ys = [] #classification 1 or 0

Xs, Ys = read_data()
print Xs[0]
Xs = numpy.array(Xs)
print "SHAPE", Xs.shape


print "Ys"
print Ys
instances_number = len(Xs)
sequence_length = len(Xs[0])
features_number = len(Xs[0][0])



a = numpy.empty((instances_number, sequence_length, features_number))


for i in range(0,instances_number):
    for j in range(0, sequence_length):
        for k in range(0, features_number):
            a[i][j][k] = Xs[i][j][k]

recall_1_list = []
recall_0_list = []
auc = []

predicted_list = []
true_list = []

numpy.random.shuffle(a)

kf = KFold(n_splits=2)

for train_idx, test_idx in kf.split(Xs):
    X_train = numpy.array([Xs[i] for i in train_idx])
    Y_train = numpy.array([Ys[i] for i in train_idx])

    X_test = numpy.array([Xs[i] for i in test_idx])
    Y_test = numpy.array([Ys[i] for i in test_idx])

    # create the model
    model = Sequential()
    model.add(Bidirectional(LSTM(100, recurrent_dropout=0.1), input_shape=(60,6)))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=3)

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
    auc.append(roc_auc_score(Y_test, rounded))

print "RESULTS:"
print "RECALL 0:", sum(recall_0_list) / float(len(recall_0_list))
print "RECALL 1:", sum(recall_1_list) / float(len(recall_1_list))
print "AUC:", sum(auc) / float(len(auc))


'''
print "RESULTS flattened:"
print "RECALL 0:", recall_score(true_list, predicted_list, pos_label=0)
print "RECALL 1:", recall_score(true_list, predicted_list, pos_label=1)
print "AUC :", roc_auc_score(true_list, predicted_list)

'''





