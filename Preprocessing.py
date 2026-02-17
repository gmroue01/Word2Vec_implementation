from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import os
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

        logger.info()
        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read() # Pour de très gros fichiers, préférez un stream
            tokens = nltk.word_tokenize(text, language=self.language)
    

  







