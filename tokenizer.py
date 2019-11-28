import nltk
import re

from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from string import punctuation

class Tokenizer:
    def __init__(self, file):
        self.file = file
        self.morth = dict()
    def tokenize(self):
        russian_stopwords = stopwords.words("russian")
        morph = MorphAnalyzer()
        tokens = nltk.word_tokenize(self.file)
        tokens = [re.sub(r'\s+', ' ', i) for i in tokens]
        tokens = [re.sub(r'[^\w\s]', ' ', i) for i in tokens]
        tokens = [token for token in tokens if token not in russian_stopwords\
                  and token != " " and token.strip() not in punctuation]
        for i in range(len(tokens)):
            if tokens[i] in self.morth:
                tokens[i] = self.morth[tokens[i]]
            else:
                self.morth[tokens[i]] = morph.parse(tokens[i])[0].normal_form
                tokens[i] = self.morth[tokens[i]]
        return tokens


file = open('codex_1.txt', 'r')
t = Tokenizer(file.read())
print(t.tokenize())

