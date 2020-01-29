#Настя

import re
import nltk
import os

from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from string import punctuation
from tools.coll import iter_by_docs

class Tokenizer:
    def __init__(self, text):
        self.text = text

    def tokenize(self, cash, morph, stop_words):
        tokens = nltk.word_tokenize(self.text)
        tokens = [re.sub(r'\s+', ' ', i) for i in tokens]
        tokens = [re.sub(r'[^\w\s]', ' ', i) for i in tokens]
        tokens = [token for token in tokens if token not in stop_words\
                  and token != " " and token.strip() not in punctuation]
        for i in range(len(tokens)):
            if tokens[i] in cash:
                tokens[i] = cash[tokens[i]]
            else:
                cash[tokens[i]] = morph.parse(tokens[i])[0].normal_form
                tokens[i] = cash[tokens[i]]
        return tokens

def docs_parser(dir):
    cash = dict()
    morph = MorphAnalyzer()
    stop_words = stopwords.words("russian")
    for file in os.listdir(dir):
        for i_d in iter_by_docs(file, dir, 'chapter', 1):
            t = Tokenizer(i_d)

'''
dir = "codexes"
docs_parser(dir)
'''
