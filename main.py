from Word2Vec import word2VecModel
from Preprocessing import preprocessing, EmbeddingIntitialiser


preprocessor = preprocessing(r"data\text\pg17989.txt", "french")
preprocessor.fit()

text_index = preprocessor.transform()




intialiser = EmbeddingIntitialiser(
    vocab_size=preprocessor.vocab_size,
    embedding_dim=100,
    seed=123
)

center_weights, context_weights = intialiser.initialize()


model = word2VecModel(text_index,center_weights,context_weights,max_iter=100,learning_rate=1e-3)

mot = text_index[0]
mot_context = text_index[1]
vect_center = center_weights[mot]
vecteur_context = context_weights[mot_context]

r = model.softmax(vecteur_context,vect_center)
print(r)