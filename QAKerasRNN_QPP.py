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
from scipy.stats.stats import pearsonr
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

    return tokens


class FastCorpusReaderYahoo:
    def __init__(self, fname):
        self.fname = fname
        self.f = open(fname, 'rb')

    def __iter__(self):
        #in Linux\Mac replace with m = mmap.mmap(self.f.fileno(), 0, prot=mmap.PROT_READ)
        m = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        data = m.readline()

        while data:

            parts = data.split(":")
            query_text =parts[1].strip().lower()
            q_ids.append(parts[0])
            data = m.readline()
            line = ExtractAlphanumeric(query_text)
            tokens = get_padded_sentences_tokens_list(line)
            line = tokens
            #Yield a list of tokens for this question
            yield line


def readY(fname):
    Ys = []
    with file(fname) as fh:
        for line in fh:
            line = line.lower()
            ap = float(line.split(",")[1])
            Ys.append(ap)
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


query_text_filename = "/Users/macbook/Desktop/corpora/QPP/query_text_clueweb_b.txt"
query_ap_filename = "/Users/macbook/Desktop/corpora/QPP/ap_clueweb.csv"

print "Read file..."
train = FastCorpusReaderYahoo(query_text_filename)

print "Creating vocab..."
vocab = Vocab.from_corpus(train)

print "Vocabulary size:", vocab.size()

WORDS_NUM = vocab.size()

embeddings_filename = "/Users/macbook/Desktop/corpora/embeddings/glove.6B.300d.txt"

print "reading embeddings"
embs = read_embeddings(embeddings_filename)

embedding_vector_length = 300

print "computing embeddings"
embedding_weights = np.zeros((WORDS_NUM, embedding_vector_length))
for word,index in vocab.w2i.items():
    if word in embs.keys():
        embedding_weights[index,:] = embs[word]
    else:
        print "NEW WORD:", word
        sampl = np.random.uniform(low=-1.0, high=1.0, size=(embedding_vector_length,))
        embedding_weights[index, :] = sampl


Ys = readY(query_ap_filename)

print "YS"
print Ys

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

accumulator_aps=[]

# fix random seed for reproducibility
np.random.seed(7)
max_sent_length = 7
kf = model_selection.KFold(n_splits=198)
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
    model.add(LSTM(100, recurrent_dropout=0.1))
    model.add(Dropout(0.8))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    #print "Fitting the model"
    #print "Model summary:"
    #print model.summary()

    model.fit(X_train, Y_train, epochs=10, batch_size=10)

    predictions = model.predict(X_test)

    for i in range(0,len(q_ids_test)):
        accumulator_aps.append(predictions[i][0])

print "Accumlator APS"
print accumulator_aps
print Ys

print "CORR COEFF", np.corrcoef(accumulator_aps,Ys)
print "Pearson and P-value",pearsonr(accumulator_aps, Ys)

