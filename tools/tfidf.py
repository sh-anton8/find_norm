# Антон

import math
import pickle
import numpy as np
from collections import Counter

class tfidf():
    def __init__(self, ttn_file, ntt_file, num_tokens_file, inv_ind_file, num_to_len_file):
        with open(ttn_file, 'rb') as f:
            self.text_to_num = pickle.load(f)
        with open(ntt_file, 'rb') as f:
            self.num_to_text = pickle.load(f)
        with open(num_tokens_file, 'rb') as f:
            self.num_to_tokens = pickle.load(f)
        with open(inv_ind_file, 'rb') as f:
            self.inv_ind = pickle.load(f)
        with open(num_to_len_file, 'rb') as f:
            self.num_to_len = pickle.load(f)
        self.tfidf = {}
        self.unique_words = set()


    def count_tf_idf(self):

        def count_tf(text):
            tf_text = Counter(text)
            for el in tf_text:
                tf_text[el] /= len(text)
            return tf_text

        def compute_idf(word, corpus):
            return math.log10(len(corpus) / len(self.inv_ind[word]))

        for key in list(self.num_to_tokens.keys()):
            tf_idf_dictionary = {}
            computed_tf = count_tf(self.num_to_tokens[key])
            for word in computed_tf:
                tf_idf_dictionary[word] = computed_tf[word] * compute_idf(word, [el[1] for el in self.num_to_tokens.items()])
            self.tfidf[key] = tf_idf_dictionary

    def save_tfidf(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self.tfidf, f)

    def take_tfidf(self, file):
        with open(file, 'rb') as f:
            self.tfidf = pickle.load(f)
        return self.tfidf

    def save_unique_words(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self.unique_words, f)

    def take_unique_words(self, file):
        with open(file, 'rb') as f:
            self.unique_words = pickle.load(f)
        return self.unique_words

    def request_counting(self, request):
        def count_tf(text):
            tf_text = Counter(text)
            for el in tf_text:
                self.unique_words.update(el)
                tf_text[el] /= len(text)
            return tf_text

        def compute_idf(word, corpus):
            if word not in self.inv_ind:
                return 0.0
            if (len(self.inv_ind[word]) != 0):
                return math.log10(len(corpus) / len(self.inv_ind[word]))
            return 0.0

        tf_idf_dictionary = {}
        computed_tf = count_tf(request)
        for word in computed_tf:
            tf_idf_dictionary[word] = computed_tf[word] * compute_idf(word, [el[1] for el in self.num_to_tokens.items()])
        return tf_idf_dictionary


''' 
mod = tfidf("/Users/shapkin/Desktop/AlgosPython/files/text_to_num_dict_codexes_for_chapter",
            "/Users/shapkin/Desktop/AlgosPython/files/num_to_text_dict_codexes_for_chapter",
            "/Users/shapkin/Desktop/AlgosPython/files/num_to_tokens_dict_codexes_for_chapter",
            "/Users/shapkin/Desktop/AlgosPython/files/inv_ind_codexes_for_chapter",
            "/Users/shapkin/Desktop/AlgosPython/files/num_to_len_dict_codexes_for_chapter")

mod.count_tf_idf()


a = mod.request_counting(['налог', 'налогооблажение', 'федеральный', 'закон', 'возврат', 'дивиденты', 'налог'])
print(a)

'''
