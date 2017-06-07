from gensim import models
import gensim
import os

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

sentences = MySentences('/Users/macbook/Desktop/corpora/triple_test') # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences, min_count=2, size=200)

print(model.most_similar(positive=['data', 'structure'], negative=['computer'], topn=1))

model.save('/Users/macbook/Desktop/corpora/mymodel')
model.save_word2vec_format('/Users/macbook/Desktop/corpora/mymodel2')
new_model = gensim.models.Word2Vec.load('/Users/macbook/Desktop/corpora/mymodel')

print model["computer"]

print(new_model.doesnt_match("query algorithm data probability ".split()))
