import numpy as np
from Preprocessing import preprocessing
class Word2Vec:

    def __init__(self,word_center,word_context):
        self.word_center = word_center
        self.word_context = word_context


    def softmax(self,u,v):
        Similarity = np.exp(u@v.T)
        
        Normalize = np.sum(np.exp(self.word_center@v.T))

        return Similarity/Normalize


    def fit(self):
        pass

    def predict(self):
        pass

path = r"./data/text/pg17989.txt" 

p = preprocessing(path, N=5)

print(p.word_center)
print(p.word_context)

word_index = p.word_index

w = Word2Vec(p.word_center,p.word_context)

index = word_index["manger"]
print(index)

u = p.word_center[index,:]
v = p.word_center[index+1,:]


r = w.softmax(u,v)

print(r)