#Настя

import re
import nltk
import os

from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from string import punctuation


class Tokenizer:
    def __init__(self, morph=None, stop_words=None):
        self.morph = morph if morph else MorphAnalyzer()
        self.stop_words = stop_words if stop_words else stopwords.words("russian")
        self.cache = {}
        
    def tokenize(self, text):
        tokens = nltk.word_tokenize(text)
        tokens = [re.sub(r'\s+', ' ', t) for t in tokens]
        tokens = [re.sub(r'[^\w\s]', ' ', t) for t in tokens]
        tokens = [t for t in tokens if t not in self.stop_words \
                  and t and t.strip() not in punctuation]
        result = []
        for t in tokens:
            if t in self.cache:
                result.append(self.cache[t])
            else:
                nf = self.morph.parse(t)[0].normal_form
                self.cache[t] = nf
                result.append(nf)
        return result
