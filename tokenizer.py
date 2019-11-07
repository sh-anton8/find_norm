import nltk
import re

from nltk.corpus import stopwords
from string import punctuation
from pymorphy2 import MorphAnalyzer

class Tokenizer:
    def __init__(self, file):
        self.file = file
    def tokenize(self):
        russian_stopwords = stopwords.words("russian")
        morph = MorphAnalyzer()
        tokens = nltk.word_tokenize(self.file)
        tokens = [re.sub(r'\s+', ' ', i) for i in tokens]
        tokens = [token for token in tokens if token not in russian_stopwords\
                  and token != " " \
                  and token.strip() not in punctuation]
        tokens = [re.sub(r'[^\w\s]', '', i) for i in tokens]
        tokens = [morph.parse(i)[0].normal_form for i in tokens]
        return tokens


