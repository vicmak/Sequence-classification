import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from collections import Counter, defaultdict
from itertools import count
import nltk
import mmap

from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from keras.layers import Dropout


q_ids = []

class Vocab: # Storing the vocabulary and word-2-id mappings
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.iteritems()}

    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(count(0).next)
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)

    def size(self): return len(self.w2i.keys())

def ExtractAlphanumeric(ins):
    from string import ascii_letters, digits, whitespace, punctuation
    return "".join([ch for ch in ins if ch in (ascii_letters + digits + whitespace + punctuation)])

def get_padded_sentences_tokens_list(text):
    tokens = []
    sentences = nltk.sent_tokenize(text)
    for sent in sentences:
        sent_tokens = nltk.word_tokenize(sent)
        tokens += ["<start>"] + sent_tokens + ["<stop>"]

    #print "SENTENCES:", sentences
    #print "TOKENS:", tokens
    return tokens


class FastCorpusReaderYahoo:
    def __init__(self, fname):
        self.fname = fname
        self.f = open(fname, 'rb')

    def __iter__(self):
        #in Linux\Mac replace with m = mmap.mmap(self.f.fileno(), 0, prot=mmap.PROT_READ)
        m = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        data = m.readline()

        description_file = "C:\\corpora\\yahoo\\descr.tsv"
        answers_file = "C:\\corpora\\yahoo\\answers.csv"

        while data:

            parts = data.split(",") #get the title of the question

            qid = parts[0] # Extract the question-ID
            q_ids.append(qid) # Add the question-ID to list of all extracted Question-IDs

            #Read the description into string (TAB separated)
            description = ""  # Init the description string
            '''
            with file(description_file) as f:
                for l in f:
                    description_parts = l.split("\t")
                    if qid == description_parts[0]:
                        description += description_parts[1]
                        #print "added:", description
            '''
            #Read the answers into string (COMMA separated)
            answer = ""
            '''
            with file(answers_file) as af:
                for l in af:
                    answer_parts = l.split(",")
                    if qid == answer_parts[0]:
                        answer += ",".join(answer_parts[1:])
            '''

            end = len(parts)-1
            text_parts = parts[1 : end] #Extract all title words, except for the classification value
            line = ",".join(text_parts)
            data = m.readline()
            line = line.lower() + description.lower() + answer.lower()
            line = ExtractAlphanumeric(line)
            #tokens = nltk.word_tokenize(line)
            tokens = get_padded_sentences_tokens_list(line)
            line = ["<start>"] + tokens + ["<stop>"]
            #Yield a list of tokens for this question
            yield line


def readY(fname):
    Ys = []
    with file(fname) as fh:
        for line in fh:
            line = line.lower()
            Ys.append(int(line.strip()[-1]))
    return Ys

def read_embeddings(embeddings_filename):
    embeddings = dict()
    with file(embeddings_filename) as f:
        for line in f:
            tokens = line.split(" ")
            word = tokens[0]
            emb = tokens[1:]
            float_emb = [float(x) for x in emb]
            embeddings[word] = float_emb
    return embeddings
'''
def writeFile(q_ids_list, predictions):

    with open("/Users/macbook/Desktop/corpora/Yahoo/result.txt", "a") as myfile:
        for i in range(0, len(q_ids_list)):
            str_r = q_ids_list[i] + " , " + str(predictions[i][0]) + "\n"
            myfile.write(str_r)
'''

title_filename = "C:\\corpora\\yahoo\\TitleUnescaped.csv"


print "Read file..."
train = FastCorpusReaderYahoo(title_filename)

print "Creating vocab..."
vocab = Vocab.from_corpus(train)

print "Vocabulary size:", vocab.size()

WORDS_NUM = vocab.size()

print vocab.w2i
print vocab.i2w

embeddings_filename = "C:\\corpora\\embeddings\\glove.6B.50d.txt"


print "reading embeddings"
embs = read_embeddings(embeddings_filename)

print embs["the"]

embedding_vector_length = 50

print "computing embeddings"
embedding_weights = np.zeros((WORDS_NUM, embedding_vector_length))
for word,index in vocab.w2i.items():
    if word in embs.keys():
        embedding_weights[index,:] = embs[word]
    else:
        sampl = np.random.uniform(low=-1.0, high=1.0, size=(50,))
        embedding_weights[index, :] = sampl

print "embedding sample"
print embedding_weights[0]
print "another embedding sample"
print embedding_weights[1]
#print "NUM of WORDS", WORDS_NUM


Ys = readY(title_filename)
train = list(train)

complete_text = ""

lengths = []


def is_common(common_list, word):
    for pair in common_list:
        if pair[0] == word:
            return True
    return False


int_train = []
i = 0

for sentence in train:
    isent = [vocab.w2i[w] for w in sentence]
    int_train.append(isent)

#print train

print len(int_train)
print len(Ys)


recall_1_list = []
recall_0_list = []
auc = []

accumulator_probs=[]

# fix random seed for reproducibility
np.random.seed(7)
max_sent_length = 50
kf = model_selection.KFold(n_splits=5)
for train_idx, test_idx in kf.split(int_train):

    X_train = [int_train[i] for i in train_idx]
    Y_train = [Ys[i] for i in train_idx]

    X_test = [int_train[i] for i in test_idx]
    Y_test = [Ys[i] for i in test_idx]
    q_ids_test = [q_ids[i] for i in test_idx]

    X_train = sequence.pad_sequences(X_train, maxlen=max_sent_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_sent_length)
    # create the model

    model = Sequential()
    model.add(Embedding(WORDS_NUM, embedding_vector_length, input_length=max_sent_length, weights=[embedding_weights]))
    model.add(Dropout(0.1))
    model.add(LSTM(300, recurrent_dropout=0.1))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print "Fitting the model"

    model.fit(X_train, Y_train, epochs=4, batch_size=100)

    predictions = model.predict(X_test)

    for i in range(0,len(q_ids_test)):
        accumulator_probs.append([q_ids_test[i], predictions[i]])

    auc.append(roc_auc_score(Y_test, predictions))

    rounded = []
    for pred in predictions:
        if pred >0.5:
            rounded.append(1)
        else:
            rounded.append(0)

    recall_0_list.append(recall_score(Y_test, rounded, pos_label=0))
    recall_1_list.append(recall_score(Y_test, rounded, pos_label=1))
    print "FINISHED FOLD - TRAIN TEXT"



def get_ensemble_data(text_probs):

    classes = []
    instances = []
    ensemble_filename = "C:\\corpora\\yahoo\\Ensemble_Data_baseline.csv"
    #print "LENGTH text probs", len(text_probs)
    for_asert = 0
    qid_counter = dict()
    with file(ensemble_filename) as f:
        for line in f:
            tokens = line.strip().split(",")
            qid = tokens[0]
            for [id, text_prob] in text_probs:
                if qid == id:
                    if qid in qid_counter.keys():
                        qid_counter[qid] += 1
                    else:
                        qid_counter[qid] = 1
                        instance =[]
                        instance.append(float(tokens[1]))
                        instance.append(text_prob[0])
                        instance.append(float(tokens[3]))
                        instances.append(instance)
                        classes.append(int(tokens[4]))
                        for_asert +=1

    return instances, classes

print "STARTING ENSEMBLE"

instances, classes = get_ensemble_data(accumulator_probs)


ensemble_recall_1_list = []
ensemble_recall_0_list = []
ensemble_auc = []

kf = model_selection.KFold(n_splits=5)

for train_idx, test_idx in kf.split(instances):

    print "ENSEMBLE FOLD"

    X_train = numpy.array([instances[i] for i in train_idx])
    Y_train = numpy.array([classes[i] for i in train_idx])

    X_test = [instances[i] for i in test_idx]
    Y_test = [classes[i] for i in test_idx]


    ensemble_model = Sequential()
    ensemble_model.add(Dense(units=3, activation="sigmoid", input_shape=(3,)))
    ensemble_model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    ensemble_model.compile(loss="binary_crossentropy", optimizer="adam")

    ensemble_model.fit(X_train, Y_train, epochs=500)

    print"FITTED"

    predictions = ensemble_model.predict(X_test)
    #print predictions

    ensemble_auc.append(roc_auc_score(Y_test, predictions))

    rounded = []
    for pred in predictions:
        if pred >0.5:
            rounded.append(1)
        else:
            rounded.append(0)

    ensemble_recall_0_list.append(recall_score(Y_test, rounded, pos_label=0))
    ensemble_recall_1_list.append(recall_score(Y_test, rounded, pos_label=1))

print "TEXT:"
print "RECALL 0:", sum(recall_0_list) / float(len(recall_0_list))
print "RECALL 1:", sum(recall_1_list) / float(len(recall_1_list))
print "AUC :", sum(auc)/float(len(auc))

print "ENSEMBLE"
print "RECALL 0:", sum(ensemble_recall_0_list) / float(len(ensemble_recall_0_list))
print "RECALL 1:", sum(ensemble_recall_1_list) / float(len(ensemble_recall_1_list))
print "AUC :", sum(ensemble_auc)/float(len(ensemble_auc))
''''''
