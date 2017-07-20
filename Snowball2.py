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
import itertools
from sklearn import model_selection
import copy
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
import random
from keras.layers import Dropout


q_ids = []
UNLABELED_INSTANCES_NUMBER = 1000


def is_common(common_list, word):
    for pair in common_list:
        if pair[0] == word:
            return True
    return False


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


vocab_filename = "C:\\corpora\\yahoo\\titles20M\\50k_vocab.txt"

def read_vocab_from_list(filename):
    w2i = dict()
    counter = 0
    with open(filename) as dict_file:
        for line in dict_file:
            word = line.strip()
            w2i[word] = counter
            counter += 1
    return w2i


vocab_dictionary = read_vocab_from_list(vocab_filename)

main_vocab = Vocab(vocab_dictionary)

print "Vocab size", main_vocab.size()

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


def get_int_sentences(sentences, vocab):
    int_sents = []
    for sentence in sentences:
        isent = [vocab.w2i[w] for w in sentence]
        int_sents.append(isent)
    return int_sents


class UnlabeledFastCorpusReader:
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        with open(self.fname) as unlabeled_file:
            for i in range(0, UNLABELED_INSTANCES_NUMBER):
                current_line = unlabeled_file.next()
                line = current_line.strip()
                tokens = line.split("\t")
                line = tokens[8]
                line = get_tokenized_padded_line(line)
                yield line


def get_tokenized_padded_line(string_line, vocab):
    line = string_line.lower()
    line = ExtractAlphanumeric(line)
    tokens = get_padded_sentences_tokens_list(line)
    line = ["<start>"] + tokens + ["<stop>"]

    clean_line = []
    for token in line:
        if token in vocab.w2i.keys():
            clean_line.append(token)
    return clean_line



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
        best_answer_file = "C:\\corpora\\yahoo\\best_answers.csv"

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

            #Read the answers into string (COMMA separated)

            '''
            with file(best_answer_file) as af:
                for l in af:
                    answer_parts = l.split(",")
                    if qid == answer_parts[0]:
                        answer += ",".join(answer_parts[1:])
            '''

            end = len(parts)-1
            text_parts = parts[1 : end] # Extract all title words, except for the classification value
            line = ",".join(text_parts)
            data = m.readline()
            line = line.lower() + description.lower() + answer.lower()
            line = get_tokenized_padded_line(line, main_vocab)
            # Yield a list of tokens for this question
            yield line


def read_next_lines(filename, from_line_number, to_line_number):

    lines = []
    with open(filename) as unlabeled_file:
        for line in itertools.islice(unlabeled_file, from_line_number, to_line_number):
            parts = line.split("\t")
            title_description = parts[1] + parts[2]
            title_description = title_description.lower()
            tokenized_title_description = get_tokenized_padded_line(title_description, main_vocab)
            lines.append(tokenized_title_description)
    return lines


def log_train_file(message):
    log_file = "C:\\corpora\\yahoo\\log_unlabeled_2.txt"
    with open(log_file, "a") as myfile:
        myfile.write(message)


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


labeled_title_filename = "C:\\corpora\\yahoo\\TitleUnescaped.csv"
#unlabeled_titles_filename = "C:\\corpora\\yahoo\\titles20M\\question.tsv"
unlabeled_titles_descriptions_filename = "C:\\corpora\\yahoo\\random_titles_descriptions.txt"
print "Read labeled file..."
labeled_train = FastCorpusReaderYahoo(labeled_title_filename)

embeddings_filename = "C:\\corpora\\embeddings\\titles300d.txt"

print "reading word2vec embeddings"
embs = read_word2vec_embeddings(embeddings_filename)

print "Checking embeddings for THE", embs["the"]

embedding_vector_length = 300

print "computing embeddings"
embedding_weights = np.zeros((main_vocab.size(), embedding_vector_length))
for word, index in main_vocab.w2i.items():
    if word in embs.keys():
        embedding_weights[index, :] = embs[word]
    else:
        sampl = np.random.uniform(low=-1.0, high=1.0, size=(embedding_vector_length,))
        embs[word] = sampl
        embedding_weights[index, :] = sampl

print "embedding sample", embedding_weights[0]
print "another embedding sample", embedding_weights[1]

print "Reading Ys"
Ys = readY(labeled_title_filename)

labeled_train = list(labeled_train)
print "Creating labeled i-sentences for training"
int_train = get_int_sentences(labeled_train, main_vocab)

unlabeled_batch_size = 2000
max_sent_length = 100

new_instances = []
new_Ys = []

original_instances = copy.deepcopy(int_train)
original_Ys = copy.deepcopy(Ys)


def get_ensemble_data(text_probs):

    classes = []
    instances = []
    ensemble_filename = "C:\\corpora\\yahoo\\Ensemble_Data_baseline.csv"
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
                        instance = []
                        instance.append(float(tokens[1]))
                        instance.append(text_prob[0])
                        instance.append(float(tokens[3]))
                        instances.append(instance)
                        classes.append(int(tokens[4]))
                        log_train_file(str(qid) + " " + str(text_prob[0]) + ", ")
    log_train_file("\n")
    return instances, classes

# Here I create a shuffled list of indexes in order not to be biased
unlabeled_instances_number = 200000
unlabeled_batches_number = unlabeled_instances_number / unlabeled_batch_size
indexes = list(range(unlabeled_batches_number))
random.shuffle(indexes)

for i in indexes:

    unlabeled_train = read_next_lines(unlabeled_titles_descriptions_filename, i*unlabeled_batch_size, i*unlabeled_batch_size + unlabeled_batch_size)
    print "Creating unlabeled i-sentences for training, batch:", i
    int_unlabeled_train = get_int_sentences(unlabeled_train, main_vocab)

    print " Building the model....."
    # start label propagation
    model = Sequential()
    model.add(Embedding(main_vocab.size(), embedding_vector_length, input_length=max_sent_length, weights=[embedding_weights]))
    model.add(Dropout(0.2))
    model.add(LSTM(200, recurrent_dropout=0.1))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print "Padding the model"
    padded_train = sequence.pad_sequences(int_train, max_sent_length)

    log_train_file("Train size: "+ str(len(padded_train)) + "\n")

    print "fitting the model"
    model.fit(padded_train, Ys, epochs=4, batch_size=100)
    print "Padding test"
    padded_test = sequence.pad_sequences(int_unlabeled_train, max_sent_length)
    print "Predicting..."
    predictions = model.predict(padded_test)

    # Here add the best predictions to the labeled instances

    new_instances_num = 0
    for i in range(0, len(predictions)):
        if predictions[i] > 0.9:
            int_train.append(int_unlabeled_train[i])
            new_instances.append(int_unlabeled_train[i])
            Ys.append(1)
            new_Ys.append(1)
            new_instances_num += 1
        if predictions[i] < 0.1:
            int_train.append(int_unlabeled_train[i])
            new_instances.append(int_unlabeled_train[i])
            Ys.append(0)
            new_Ys.append(0)
            new_instances_num += 1
    print "Added new instances:", new_instances_num
    log_train_file("Added new instances: " + str(new_instances_num) + "\n")
    print "STARTING K-FOLD, TEST"

    recall_1_list = []
    recall_0_list = []
    auc = []
    accumulator_probs = []

    kf = model_selection.KFold(n_splits=5)
    for train_idx, test_idx in kf.split(original_instances):

        X_train = [original_instances[i] for i in train_idx]
        Y_train = [original_Ys[i] for i in train_idx]
        X_train = X_train + new_instances
        Y_train = Y_train + new_Ys

        X_test = [int_train[i] for i in test_idx]
        Y_test = [Ys[i] for i in test_idx]
        q_ids_test = [q_ids[i] for i in test_idx]

        X_train = sequence.pad_sequences(X_train, maxlen=max_sent_length)
        X_test = sequence.pad_sequences(X_test, maxlen=max_sent_length)

        model = Sequential()
        model.add(Embedding(main_vocab.size(), embedding_vector_length, input_length=max_sent_length, weights=[embedding_weights]))
        model.add(LSTM(200, recurrent_dropout=0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, Y_train, epochs=4, batch_size=100)
        predictions = model.predict(X_test)

        for i in range(0, len(q_ids_test)):
            accumulator_probs.append([q_ids_test[i], predictions[i]])

        auc.append(roc_auc_score(Y_test, predictions))

        rounded = []
        for pred in predictions:
            if pred > 0.5:
                rounded.append(1)
            else:
                rounded.append(0)

        recall_0_list.append(recall_score(Y_test, rounded, pos_label=0))
        recall_1_list.append(recall_score(Y_test, rounded, pos_label=1))
        print "FINISHED FOLD - TRAIN TEXT"

    print "STARTING ENSEMBLE"

    instances, classes = get_ensemble_data(accumulator_probs)



    ensemble_recall_1_list = []
    ensemble_recall_0_list = []
    ensemble_auc = []

    kf = model_selection.KFold(n_splits=5)


    for train_idx, test_idx in kf.split(instances):

        print "ENSEMBLE FOLD"

        X_train = np.array([instances[i] for i in train_idx])
        Y_train = np.array([classes[i] for i in train_idx])

        X_test = [instances[i] for i in test_idx]
        Y_test = [classes[i] for i in test_idx]

        ensemble_model = Sequential()
        ensemble_model.add(Dense(units=3, activation="sigmoid", input_shape=(3,)))
        ensemble_model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

        ensemble_model.compile(loss="binary_crossentropy", optimizer="adam")

        ensemble_model.fit(X_train, Y_train, epochs=300)

        print"FITTED"

        predictions = ensemble_model.predict(X_test)
        # print predictions

        ensemble_auc.append(roc_auc_score(Y_test, predictions))

        rounded = []
        for pred in predictions:
            if pred > 0.5:
                rounded.append(1)
            else:
                rounded.append(0)

        ensemble_recall_0_list.append(recall_score(Y_test, rounded, pos_label=0))
        ensemble_recall_1_list.append(recall_score(Y_test, rounded, pos_label=1))

    print "TEXT:"
    print "RECALL 0:", sum(recall_0_list) / float(len(recall_0_list))
    print "RECALL 1:", sum(recall_1_list) / float(len(recall_1_list))
    print "AUC :", sum(auc) / float(len(auc))

    print "ENSEMBLE"
    print "RECALL 0:", sum(ensemble_recall_0_list) / float(len(ensemble_recall_0_list))
    print "RECALL 1:", sum(ensemble_recall_1_list) / float(len(ensemble_recall_1_list))
    print "AUC :", sum(ensemble_auc) / float(len(ensemble_auc))

    log_train_file("TEXT R0: " + str(sum(recall_0_list) / float(len(recall_0_list))) + "\n")
    log_train_file("TEXT R1: " + str(sum(recall_1_list) / float(len(recall_1_list))) + "\n")
    log_train_file("TEXT AUC: " + str(sum(auc) / float(len(auc))) + "\n")

    log_train_file("ENSEMBLE R0: " + str(sum(ensemble_recall_0_list) / float(len(ensemble_recall_0_list))) + "\n")
    log_train_file("ENSEMBLE R1: " + str(sum(ensemble_recall_1_list) / float(len(ensemble_recall_1_list))) + "\n")
    log_train_file("ENSEMBLE AUC: " + str(sum(auc) / float(len(auc))) + "\n")




