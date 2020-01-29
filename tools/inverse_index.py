# Антон

import os
import tools.coll as coll
import pickle
from pymorphy2 import MorphAnalyzer
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
        self.num_tokens_dict = {}
        self.num_to_len = {}
        self.num_to_name = {}

    def update_dicts(self, par):  # par - по чему итерируемся, например: 'paragraph' (подробнее в coll)
        for file in os.listdir(self.dir):
            d1, d2 = coll.iter_by_docs(file, self.dir, par, 0)
            self.num_to_text.update(d1)
            self.text_to_num.update(d2)
            self.num_to_name.update(coll.iter_by_docs(file, self.dir, 'art_name', 1))


    def num_tokens_dict_builder(self):
        for key in list(self.num_to_text.keys()):
            self.tokenizer.text = self.num_to_text[key]
            tokens = self.tokenizer.tokenize(self.cash, self.morph, self.stop_words)
            self.num_tokens_dict[key] = tokens

    def build_inversed_index(self, par):  # par - по чему итерируемся, например: 'paragraph'
        for file in os.listdir(self.dir):
            for i_d in coll.iter_by_docs(file, self.dir, par, 1):
                self.tokenizer.text = i_d
                tokens = self.tokenizer.tokenize(self.cash, self.morph, self.stop_words)
                for token in tokens:
                    if token in self.inv_ind:
                        if (self.text_to_num[i_d], tokens.count(token)) not in self.inv_ind[token]:
                            self.inv_ind[token].append((self.text_to_num[i_d], tokens.count(token)))
                    else:
                        self.inv_ind[token] = [(self.text_to_num[i_d], tokens.count(token))]
                self.num_to_len[self.text_to_num[i_d]] = len(list(tokens))



def save_inverse_index(inv_ind, file):
    with open(file, 'wb') as f:
        pickle.dump(inv_ind, f)


def load_inverse_index(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
