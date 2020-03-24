import gensim.summarization.bm25 as bm
from tools.inverse_index import InvIndex
import pickle

PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25

class My_BM:
    def __init__(self, inverse_index_pickle):
        self.inverse_index = InvIndex.load(inverse_index_pickle)
        self.corpus = []
        self.ind_to_num_dict = {}
        self.num_to_ind_dict = {}

        ind = 0
        for key in list(self.inverse_index.num_tokens_dict.keys()):
            self.corpus.append(" ".join(self.inverse_index.num_tokens_dict[key]))
            self.ind_to_num_dict[ind] = key
            self.num_to_ind_dict[key] = ind
            ind += 1

        self.bm_oject = bm.BM25(self.corpus)

    def get_feature(self, request_text, article_num):
        return self.bm_oject.get_score(request_text, self.num_to_ind_dict[article_num])

    # сохраняем весь объект
    def save(self, file):
        print('Saving BM25 to: {}'.format(file))
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    # загружаем весь объект, использовать можно потом только его часть
    # для этого используем модификатор @staticmethod
    @staticmethod
    def load(file):
        print('Loading BM25 from: {}'.format(file))
        with open(file, 'rb') as f:
            return pickle.load(f)
