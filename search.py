# Антон

import os
import numpy as np
import pickle
import inverse_index as ii
import tokenize_docs
from pymorphy2 import MorphAnalyzer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import request_recognizer


def glue_func(array):
    ans = array[0]
    for i in range(1, len(array)):
        ans += ' ' + array[i]
    return ans


class Search:
    def __init__(self, tokenizer):
        self.inv_ind = {}
        self.tokenizer = tokenizer
        self.text_to_num = {}
        self.num_to_text = {}
        self.cash = dict()
        self.morph = MorphAnalyzer()
        self.stop_words = stopwords.words("russian")

    def init_index_and_dicts(self, inv_ind_file, ttn_dict_file, ntt_dict_file):
        with open(ttn_dict_file, 'rb') as f:
            self.text_to_num = pickle.load(f)
        with open(ntt_dict_file, 'rb') as f:
            self.num_to_text = pickle.load(f)
        with open(inv_ind_file, 'rb') as f:
            self.inv_ind = pickle.load(f)

    def request_processing_input(self):
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

    def request_processing_pravoved(self, directory):
        s = request_recognizer.Separator(directory)
        requests = s.sep_by_requests()
        for reqst in requests:
            self.tokenizer.text = reqst.question
            rqst = self.tokenizer.tokenize(self.cash, self.morph, self.stop_words)
            ans = Counter()
            for word in rqst:
                if word in self.inv_ind:
                    for el in self.inv_ind[word]:
                        ans[el] += 1
            top = ans.most_common(5)
            i = 1
            for el in top:
                if (i < 2):
                    print(reqst.question)
                print()
                print(i, '-ый по релевантности ответ: ', el[0], sep='')
                print("Текст статьи (первые 200 символов): ")
                print(self.num_to_text[el[0]][:200])
                i += 1
                if (i > 6):
                    print()



# dir = "codexes"
# s = ii.InvIndex(dir, tokenize_docs.Tokenizer('d'))
# s.update_dicts('paragraph')
# s.build_inversed_index('paragraph')
# s.save_index("inv_ind")
# s.save_ntt_dict("ntt_dict")
# s.save_ttn_dict("ttn_dict")

# searcher = Search(tokenize_docs.Tokenizer('d'))
# searcher.init_index_and_dicts("inv_ind", "ttn_dict", "ntt_dict")
# searcher.request_processing_input()
# dir2 = "pravoved_articles.txt"
# searcher.request_processing_pravoved(dir2)











