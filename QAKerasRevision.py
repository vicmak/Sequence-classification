import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Bidirectional
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from collections import Counter, defaultdict
from itertools import count
import nltk
import mmap
from keras import backend as K
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
        tokens += ["<sentence-start>"] + sent_tokens + ["<sentence-stop>"]

    return tokens


class FastCorpusReaderYahoo:
    def __init__(self, fname):
        self.fname = fname
        self.f = open(fname, 'rb')

    def __iter__(self):
        #in Linux\Mac replace with m = mmap.mmap(self.f.fileno(), 0, prot=mmap.PROT_READ)
        m = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        data = m.readline()

        description_file = "/Users/macbook/Desktop/corpora/Yahoo/descr.tsv"
        answers_file = "/Users/macbook/Desktop/corpora/Yahoo/AllanswersUnescaped.csv"
        best_answer_filename = "/Users/macbook/Desktop/corpora/Yahoo/BestanswerUnescaped.csv"

        while data:

            parts = data.split(",") #get the title of the question

            qid = parts[0]  # Extract the question-ID
            q_ids.append(qid)  # Add the question-ID to list of all extracted Question-IDs

            # Read the description into string (TAB separated)
            description = ""  # Init the description string
            answer = ""

            with file(description_file) as f:
                for l in f:
                    description_parts = l.split("\t")
                    if qid == description_parts[0]:
                        description += description_parts[1]
                        #print "added:", description

            # Read the answers into string (COMMA separated)


            with file(best_answer_filename) as af:
                for l in af:
                    answer_parts = l.split(",")
                    if qid == answer_parts[0]:
                        answer += ",".join(answer_parts[1:])

            end = len(parts)-1
            text_parts = parts[1: end]  # Extract all title words, except for the classification value
            line = ",".join(text_parts)
            data = m.readline()
            line = line.lower() + description.lower() + answer.lower()
            line = ExtractAlphanumeric(line)
            tokens = get_padded_sentences_tokens_list(line)
            line = ["<start>"] + tokens + ["<stop>"]
            # Yield a list of tokens for this question
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
            word = tokens[0].strip()
            emb = tokens[1:]
            float_emb = [float(x) for x in emb]
            embeddings[word] = float_emb
    return embeddings


def read_word2vec_embeddings(embeddings_filename):
    embeddings = dict()
    counter = 0
    with file(embeddings_filename) as f:
        for line in f:
            if counter > 0:
                tokens = line.split(" ")
                word = tokens[0]
                emb = tokens[1:]
                float_emb = [float(x) for x in emb]
                embeddings[word] = float_emb
            counter += 1
    return embeddings


title_filename = "/Users/macbook/Desktop/corpora/Yahoo/TitleUnescaped.csv"


print "Read file..."
train = FastCorpusReaderYahoo(title_filename)

print "Creating vocab..."
vocab = Vocab.from_corpus(train)

print "Vocabulary size:", vocab.size()

WORDS_NUM = vocab.size()


embeddings_filename = "/Users/macbook/Desktop/corpora/embeddings/titles300d.txt"
#embeddings_filename = "/Users/macbook/Desktop/corpora/embeddings/glove.6B.50d.txt"

print "Using the following embeddings", embeddings_filename

print "reading word2vec embeddings"
embs = read_word2vec_embeddings(embeddings_filename)
#embs = read_embeddings(embeddings_filename)

print embs["the"]

embedding_vector_length = 300

print "computing embeddings"
embedding_weights = np.zeros((WORDS_NUM, embedding_vector_length))
for word,index in vocab.w2i.items():
    if word in embs.keys():
        embedding_weights[index,:] = embs[word]
    else:
        sampl = np.random.uniform(low=-1.0, high=1.0, size=(embedding_vector_length,))
        embs[word] = sampl
        embedding_weights[index, :] = sampl

print "embedding sample"
print embedding_weights[0]
print "another embedding sample"
print embedding_weights[1]


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


print "len train", len(int_train)
print "len ys", len(Ys)


recall_1_list = []
recall_0_list = []
auc = []

accumulator_probs = []

# fix random seed for reproducibility
np.random.seed(7)
max_sent_length = 800
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
    model.add(LSTM(800))
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


print "TEXT:"
print " RECALL 0:", sum(recall_0_list) / float(len(recall_0_list)), "RECALL 0 STD:", np.std(recall_0_list)
print " RECALL 1:", sum(recall_1_list) / float(len(recall_1_list)), "RECALL 1 STD:", np.std(recall_1_list)
print " AUC :", sum(auc)/float(len(auc)), "AUC STD:", np.std(auc)


