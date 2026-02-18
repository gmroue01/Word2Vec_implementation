from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import os
import logger
from typing import List, Dict, Tuple, Generator
nltk.download('punkt')
nltk.download('punkt_tab')
class preprocessing:

    def __init__(self,file_path,language):
        self.file_path = file_path
        self.language = language
        
        self.vocab : List[str] = []
        self.word_to_idx : Dict[str,int] = {}


    def fit(self) -> None:
        """
        Read a file and build the token list

        :param self: Description
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Le fichier {self.file_path} est introuvable")

        
        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read() # Pour de très gros fichiers, préférez un stream
            tokens = nltk.word_tokenize(text, language=self.language)

        self.vocab() = sorted(list(set(tokens)))
        
        self.word_to_idx = {word : idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx : word for word, idx in enumerate(self.vocab)}
        
        logger.info(f"Voc list built. Size :"{len(self.vocab)})

  







