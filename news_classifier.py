import numpy
from keras.models import Sequential

from keras.layers import Dense
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from collections import Counter, defaultdict
from itertools import count
import nltk
import mmap
import os
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from keras.layers import Dropout


class Vocab:  # Storing the vocabulary and word-2-id mappings
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i: w for w, i in w2i.iteritems()}

    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(count(0).next)
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)

    def size(self): return len(self.w2i.keys())


def ExtractAlphanumeric(ins):
    from string import ascii_letters, digits, whitespace, punctuation
    return "".join([ch for ch in ins if ch in (ascii_letters + whitespace + punctuation)])


def get_padded_sentences_tokens_list(text, mark=""):
    tokens = []
    sentences = nltk.sent_tokenize(text)
    for sent in sentences:
        sent_tokens = nltk.word_tokenize(sent)
        new_tokens = [token + mark for token in sent_tokens]
        tokens += ["<start>"] + new_tokens + ["<stop>"]

    return tokens


class NewsCorpusReader:
    def __init__(self, positive_news_path, negative_news_path):
        self.positive_news_path = positive_news_path
        self.negative_news_path = negative_news_path
        self.positive_number = 0
        self.negative_number = 0

    def __iter__(self): # Yields one instance as a list of words

        for root, dirs, files in os.walk(self.positive_news_path):
            path = root.split(os.sep)
            print((len(path) - 1) * '---', os.path.basename(root))
            for file in files:

                # print(len(path) * '---', file)

                title = file.lower()

                title = ExtractAlphanumeric(title)

                title_tokens = get_padded_sentences_tokens_list(title)

                tokens_list = ["<start>"] + title_tokens + ["<stop>"]
                # Yield a list of tokens for this question
                self.positive_number += 1
                yield tokens_list

        for root, dirs, files in os.walk(self.negative_news_path):
            path = root.split(os.sep)
            print((len(path) - 1) * '---', os.path.basename(root))
            for file in files:

                # print(len(path) * '---', file)

                title = file.lower()

                title = ExtractAlphanumeric(title)

                title_tokens = get_padded_sentences_tokens_list(title)

                tokens_list = ["<start>"] + title_tokens + ["<stop>"]
                # Yield a list of tokens for this question
                self.negative_number += 1
                yield tokens_list

print "Read file..."
train = NewsCorpusReader("/Users/macbook/Dropbox/EventRegistry/Times of Israel", "/Users/macbook/Dropbox/EventRegistry/www.aljazeera.com")

print "Creating vocab..."
vocab = Vocab.from_corpus(train)

positive_train_num = train.positive_number
negative_train_num = train.negative_number

print "Vocabulary size:", vocab.size()

train_list = list(train)


print "all", len(train_list)


positive_examples = train_list[0:positive_train_num]
negative_examples = train_list[positive_train_num:]

train_list = [j for i in zip(positive_examples, negative_examples) for j in i]


Ys = []
for i in range(0, positive_train_num):
    Ys.append(1)
for i in range(0, negative_train_num):
    Ys.append(0)

positive_labels = Ys[0:positive_train_num]
negative_labels = Ys[positive_train_num:]

Ys = [j for i in zip(positive_labels, negative_labels) for j in i]

int_train = []

for sentence in train_list:
    print "sentence:",sentence
    isent = [vocab.w2i[w] for w in sentence]
    int_train.append(isent)

print "total instances:", len(int_train)


numpy.random.seed(7)
max_sent_length = 10
embedding_vector_length = 50
WORDS_NUM = vocab.size()


kf = model_selection.KFold(n_splits=5)
for train_idx, test_idx in kf.split(int_train):

    X_train = [int_train[i] for i in train_idx]
    Y_train = [Ys[i] for i in train_idx]

    X_test = [int_train[i] for i in test_idx]
    Y_test = [Ys[i] for i in test_idx]

    X_train = sequence.pad_sequences(X_train, maxlen=max_sent_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_sent_length)
    # create the model

    model = Sequential()
    model.add(Embedding(WORDS_NUM, embedding_vector_length, input_length=max_sent_length))

    model.add(LSTM(50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print "Fitting the model"

    model.fit(X_train, Y_train, epochs=4, batch_size=100)

    predictions = model.predict(X_test)

    print "AUC:", roc_auc_score(Y_test, predictions)




