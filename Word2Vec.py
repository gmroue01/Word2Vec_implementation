import numpy as np
from Preprocessing import preprocessing, EmbeddingIntitialiser


class word2VecModel:

    def __init__(self, text_index, word_center, word_context,max_iter,learning_rate):
        self.text_index = text_index
        self.word_center = word_center
        self.word_context = word_context

        self.loss_history = None
        self.max_iter = max_iter
        self.learning_rate = learning_rate
    

    def softmax(self, u, v):
        Similarity = np.exp(u@v)

        Normalize = np.sum(np.exp(self.word_center@v))

        return Similarity/Normalize

    def loss(self,y,t):
        """
        Loss : y - t
        
        :param self: Description
        :param y: Vecteur de probabilité P(wt+j|wt)
        :param t: Index du mot cible
        """
        loss_ = y[t] - 1
        self.loss_history.append(loss_)
        return loss_
    

     


    def fit(self):

        for i in range(self.max_iter):
            
            pass

    def predict(self):
        pass


path = r"./data/text/pg17989.txt"

preprocessor = preprocessing(path, "french")
preprocessor.fit()
text_index = preprocessor.transform()

initialiser = EmbeddingIntitialiser(
    preprocessor.vocab_size,
    embedding_dim=100,
    seed=442
)

word_center, word_context = initialiser.initialize()



