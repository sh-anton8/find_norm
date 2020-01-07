# Антон

import os
import tools.coll as coll
import numpy as np
import pickle
import tools.tokenize_docs as tokenize_docs
from pymorphy2 import MorphAnalyzer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords



class InvIndex():
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

    def save_ttn_dict(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self.text_to_num, f)

    def take_ttn_dict(self, file):
        with open(file, 'rb') as f:
            self.text_to_num = pickle.load(f)
        return self.text_to_num

    def save_ntt_dict(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self.num_to_text, f)

    def take_ntt_dict(self, file):
        with open(file, 'rb') as f:
            self.num_to_text = pickle.load(f)
        return self.num_to_text

    def save_index(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self.inv_ind, f)

    def take_index_from_file(self, file):
        with open(file, 'rb') as f:
            self.inv_ind = pickle.load(f)
        return self.inv_ind

    def build_inversed_index(self, par): # par - по чему итерируемся, например: 'paragraph'
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
