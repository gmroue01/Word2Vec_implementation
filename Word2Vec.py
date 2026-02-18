import numpy as np
from Preprocessing import preprocessing, EmbeddingIntitialiser


class Word2Vec:

    def __init__(self, word_center, word_context):
        self.word_center = word_center
        self.word_context = word_context

    def softmax(self, u, v):
        Similarity = np.exp(u@v)

        Normalize = np.sum(np.exp(self.word_center@v))

        return Similarity/Normalize

    def loss(self):
        


    def fit(self):
        pass

    def predict(self):
        pass


path = r"./data/text/pg17989.txt"

preprocessor = preprocessing(path, "french")
preprocessor.fit()

initialiser = EmbeddingIntitialiser(
    preprocessor.vocab_size,
    embedding_dim=100,
    seed=442
)
print(preprocessor.vocab)
word_center, word_context = initialiser.initialize()


index_u = preprocessor.word_to_idx['je']
index_v = preprocessor.word_to_idx['suis']
u = word_center[index_u, :]
v = word_context[index_v, :]

w = Word2Vec(word_center, word_context)

r = w.softmax(u, v)


