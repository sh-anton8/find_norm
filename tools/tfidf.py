# Антон

import math
import pickle
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import tools.inverse_index as ii

class tfidf():
    def __init__(self, inverse_index_pickle, ngramm=(1, 1)):
        self.vectorizer = TfidfVectorizer(preprocessor=None, encoding='utf-8', ngram_range=ngramm)
        self.tfidf_matrix = np.array([])
        self.num_to_num_dict = {}
        self.inverse_index = ii.load_inverse_index(inverse_index_pickle)

    def count_tf_idf(self):
        corpus = []
        ind = 0
        for key in list(self.inverse_index.num_tokens_dict.keys()):
            corpus.append(" ".join(self.inverse_index.num_tokens_dict[key]))
            self.num_to_num_dict[ind] = key
            ind += 1
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def request_counting(self, request):
        request_tfidf = self.vectorizer.transform(request)
        return request_tfidf


def save_tfidf(tfidf, file):
    with open(file, 'wb') as f:
        pickle.dump(tfidf, f)


def load_tfidf(file):
    with open(file, 'rb') as f:
        return pickle.load(f)