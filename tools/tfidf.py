# Антон

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tools.inverse_index import InvIndex
from tqdm import tqdm


class TFIDF:
    def __init__(self, inverse_index_pickle, ngramm=(1, 1), norm='l2', use_idf=True,
                 sublinear_tf=False):
        self.vectorizer = TfidfVectorizer(preprocessor=None, encoding='utf-8', ngram_range=ngramm,
                                          norm=norm, use_idf=use_idf, sublinear_tf=sublinear_tf)
        self.tfidf_matrix = np.array([])
        self.num_to_num_dict = {}
        self.inverse_index = InvIndex.load(inverse_index_pickle)

    def count_tf_idf(self):
        corpus = []
        ind = 0
        ta = tqdm(total=len(list(self.inverse_index.num_tokens_dict.keys())))
        for key in list(self.inverse_index.num_tokens_dict.keys()):
            corpus.append(" ".join(self.inverse_index.num_tokens_dict[key]))
            self.num_to_num_dict[ind] = key
            ind += 1
            ta.update(1)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        ta.close()

    def request_counting(self, request):
        request_tfidf = self.vectorizer.transform(request)
        return request_tfidf

    # сохраняем весь объект
    def save(self, file):
        print('Saving tf-idf to: {}'.format(file))
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    # загружаем весь объект, использовать можно потом только его часть
    # для этого используем модификатор @staticmethod
    @staticmethod
    def load(file):
        print('Loading tfidf from: {}'.format(file))
        with open(file, 'rb') as f:
            return pickle.load(f)
