# Антон

import os
import coll
import numpy as np
import pickle
import tokenize_docs
from pymorphy2 import MorphAnalyzer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


class Search:
    def __init__(self, directory, tokenizer):
        self.dir = directory
        self.inv_ind = {}
        self.tokenizer = tokenizer
        self.text_to_num = {}
        self.num_to_text = {}
        self.cash = dict()
        self.morph = MorphAnalyzer()
        self.stop_words = stopwords.words("russian")


    def update_dicts(self, par): # par - по чему итерируемся, например: 'paragraph'
        for file in os.listdir(self.dir):
            d1, d2 = coll.iter_by_docs(file, self.dir, par, 0)
            self.num_to_text.update(d1)
            self.text_to_num.update(d2)

    def inversed_index(self, par): # par - по чему итерируемся, например: 'paragraph'
        for file in os.listdir(self.dir):
            for i_d in coll.iter_by_docs(file, self.dir, par, 1):
                self.tokenizer.text = i_d
                tokens = self.tokenizer.tokenize(self.cash, self.morph, self.stop_words)
                for token in tokens:
                    if token in self.inv_ind:
                        if self.text_to_num[i_d] not in self.inv_ind[token]:
                            self.inv_ind[token].append(self.text_to_num[i_d])
                    else:
                        self.inv_ind[token] = [self.text_to_num[i_d]]
        return self.inv_ind

    def save_index(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self.inv_ind, f)

    def take_index_from_file(self, file):
        with open(file, 'rb') as f:
            self.inv_ind = pickle.load(f)
        return self.inv_ind

    def request_processing(self):
        reqst = input('Введите запрос: ')
        self.tokenizer.text = reqst
        reqst = self.tokenizer.tokenize(self.cash, self.morph, self.stop_words)
        ans = Counter()
        for word in reqst:
            if word in self.inv_ind:
                for el in self.inv_ind[word]:
                    ans[el] += 1
        top = ans.most_common(5)
        i = 1
        for el in top:
            print(i, '-ый по релевантности ответ: ', el[0], sep='')
            print("Текст статьи: ")
            print(self.num_to_text[el[0]])
            print()
            print()
            i += 1



# dir = "codexes"
# s = Search(dir, tokenize_docs.Tokenizer('d'))
# s.update_dicts('paragraph')
# s.inversed_index('paragraph')
# s.save_index("inv_ind")
# s.take_index_from_file("inv_ind")
# s.request_processing()









