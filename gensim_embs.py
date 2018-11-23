from gensim import models
import gensim
import os


def ExtractAlphanumeric(ins):
    from string import ascii_letters, digits, whitespace, punctuation
    return "".join([ch for ch in ins if ch in (ascii_letters + digits + whitespace + punctuation)])

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for root, dirs, files in os.walk(self.dirname):
            path = root.split(os.sep)
            print "Path:", path
            for filename in files:
                full_path = root + "/" + filename
                for line in open(full_path):
                    line = ExtractAlphanumeric(line)
                    yield line.split()

    #def __iter__(self):
    #    for fname in os.listdir(self.dirname):
    #        for line in open(os.path.join(self.dirname, fname)):
    #            yield line.split()

sentences = MySentences('/Users/macbook/Desktop/corpora/news/vocabfiles') # a memory-friendly iterator

print "Start training.."

model = gensim.models.Word2Vec(sentences, min_count=5, size=300, workers=4, window=5, negative=5)

print "Saving model"

model.wv.save_word2vec_format('/Users/macbook/Desktop/corpora/embeddings/news300d.txt')

print "End training"
#new_model = gensim.models.Word2Vec.load('/Users/macbook/Desktop/corpora/mymodel')


#print(new_model.doesnt_match("query algorithm data probability ".split()))