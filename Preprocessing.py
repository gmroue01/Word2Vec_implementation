from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import os
import logger
import logging
from typing import List, Dict, Tuple, Generator
nltk.download('punkt')
nltk.download('punkt_tab')


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class preprocessing:

    def __init__(self, file_path, language):
        self.file_path = file_path
        self.language = language

        self.vocab: List[str] = []
        self.word_to_idx: Dict[str, int] = {}

    def fit(self) -> None:
        """
        Read a file and build the token list

        :param self: Description
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Le fichier {self.file_path} est introuvable")

        logger.info(f"Processing of the file : {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read().lower()  # Pour de très gros fichiers, préférez un stream
            tokens = nltk.word_tokenize(text, language=self.language)

        self.vocab = sorted(list(set(tokens)))

        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}

        logger.info(f"Voc list built. Size :{len(self.vocab)}")

    def transform(self):
        """
        Transform the text into a array of index
        
        :param self: Description
        """
        text_index = []
        for word in self.tokens:
            text_index.append(self.word_to_idx[word])
        return text_index
    

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


class EmbeddingIntitialiser:
    """
    Initiate the weight of the model
    """

    def __init__(self, vocab_size: int, embedding_dim: int, seed: int = 42):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def initialize(self) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(
            f"Initialisation of embedding vectors (Dim:{self.embedding_dim})")

        word_center = self.rng.standard_normal(
            (self.vocab_size, self.embedding_dim))
        word_context = self.rng.standard_normal(
            (self.vocab_size, self.embedding_dim))

        return word_center, word_context


# preprocessor = preprocessing(r"data\text\pg17989.txt", "french")
# preprocessor.fit()

# intialiser = EmbeddingIntitialiser(
#     vocab_size=preprocessor.vocab_size,
#     embedding_dim=100,
#     seed=123
# )

# center_weights, context_weights = intialiser.initialize()
