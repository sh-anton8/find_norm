from tools.inverse_index import InvIndex
from tools.bm25 import My_BM
import os
from tools.pravoved_recognizer import Request
from tools.tfidf import TFIDF
import typing as tp
from tools.tokenize_docs import Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
from pymorphy2 import MorphAnalyzer
from collections import Counter
from nltk.corpus import stopwords
from tools import coll

from tools.relative_paths_to_directories import path_to_directories

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25, \
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())

'''
Реализация класса для подсчета всех признаков для машинного обучения
'''


class Features:
    def __init__(self, inv_ind_pickle, bm_25_file):
        self.inv_ind = InvIndex.load(inv_ind_pickle)
        self.inv_in_path = inv_ind_pickle
        self.cash = dict()
        self.morph = MorphAnalyzer()
        self.stop_words = stopwords.words("russian")
        self.tfidfs = []
        self.art_names = {}
        self.first_tf_idf = None
        if not os.path.isfile(bm_25_file):
            self.bm_obj = My_BM(PATH_TO_INV_IND)
            self.bm_obj.save(bm_25_file)
        else:
            self.bm_obj = My_BM.load(bm_25_file)

    def get_fmerRelev_feature(self, req_text, article_num):
        # Считается F-мера для запроса и номера статьи в нашей нумерации (кодекс, статья)
        reqst = req_text
        self.inv_ind.tokenizer.text = reqst
        reqst = self.inv_ind.tokenizer.tokenize(self.inv_ind.cash, self.inv_ind.morph, self.inv_ind.stop_words)
        help = Counter()
        for word in reqst:
            if word in self.inv_ind.inv_ind:
                for key in self.inv_ind.inv_ind[word]:
                    help[key[0]] += 1
        Recall = help[article_num] / len(reqst)
        if not self.inv_ind.num_tokens_dict[article_num]:
            F = -1000000000
        else:
            Presicion = help[article_num] / len(self.inv_ind.num_tokens_dict[article_num])
            if not help[article_num]:
                F = -1000000000
            else:
                F = 2 / ((1 / Presicion) + (1 / Recall))
        return F

    def get_doc_len_feature(self, req_text, article_num):
        # Считается длина документа
        if article_num not in self.inv_ind.num_to_len:
            return 0

        return self.inv_ind.num_to_len[article_num]

    def get_bm25_feature(self, req_text, article_num):
        # Считается bm25 для поданной на вход статьи и запроса
        return self.bm_obj.get_feature(req_text, article_num)

    def _tfidf_cnt(self, ngramm: tp.Tuple[int, int], inv_ind_path: str, tfidf_path: str, num: int,
                  norm: str = 'l2', use_idf: bool = True, sublinear_tf: bool = False) -> None:
        # считает tfidf по корпусу с параметрами
        mod = TFIDF(inv_ind_path, ngramm=ngramm, norm=norm, use_idf=use_idf, sublinear_tf=sublinear_tf)
        mod.count_tf_idf()
        mod.save(tfidf_path + '_' + str(num))

    def _if_file_not_exist(self, tfidf_path, num):
        # проверяет существует ли файл
        if not os.path.exists(tfidf_path + f'_{num}'):
            return True
        return False

    def _count_tf_idf(self):
        # считает все tfidf, чтобы впоследствии по ним посчитать косинусовую меру
        self.path_to_tf_idf = os.path.join(PATH_TO_TF_IDF, 'tf_idf')
        if self._if_file_not_exist(self.path_to_tf_idf, 1):
            self._tfidf_cnt((1, 1), self.inv_in_path, self.path_to_tf_idf, 1)
        if self._if_file_not_exist(self.path_to_tf_idf, 2):
            self._tfidf_cnt((3, 3), self.inv_in_path , self.path_to_tf_idf, 2)
        if self._if_file_not_exist(self.path_to_tf_idf, 3):
            self._tfidf_cnt((1, 3), self.inv_in_path , self.path_to_tf_idf, 3)
        if self._if_file_not_exist(self.path_to_tf_idf, 4):
            self._tfidf_cnt((1, 1), self.inv_in_path, self.path_to_tf_idf, 4, norm='l1', use_idf=False)
        if self._if_file_not_exist(self.path_to_tf_idf, 5):
            self._tfidf_cnt((1, 1), self.inv_in_path, self.path_to_tf_idf, 5, use_idf=False, sublinear_tf=True)
        if self._if_file_not_exist(self.path_to_tf_idf, 6):
            self._tfidf_cnt((1, 1), self.inv_in_path, self.path_to_tf_idf, 6, sublinear_tf=True)

    def load_all_tfidf(self):
        # считает все tf-idf, если они еще не посчитаны
        # сохраняет все загруженные tf-idf в массив self.tfidfs
        self._count_tf_idf()
        for i, file in enumerate(sorted(os.listdir(PATH_TO_TF_IDF))):
            tfidf_cnt = TFIDF.load(os.path.join(PATH_TO_TF_IDF, file))
            if i == 1:
                self.first_tf_idf = tfidf_cnt
            self.tfidfs.append(tfidf_cnt)

    def features_cos_sim(self, req: Request):
        # считает косинусиновую меру для заданного запроса и всех tf_idf
        cos_simil = [0] * 6
        for i in range(6):
            cos_simil[i] = self._count_cos_similarity(req, self.tfidfs[i])
        return cos_simil

    def _count_cos_similarity(self, req: Request, tfidf_loaded):
        # считает косинусиновую меру для заданного запроса и одного заданного tf_idf
        t = Tokenizer(req.question)
        req = t.tokenize(self.cash, self.morph, self.stop_words)
        req_tfidf_dict = tfidf_loaded.request_counting([" ".join(req)])
        cos_sim = cosine_similarity(tfidf_loaded.tfidf_matrix, req_tfidf_dict)
        return cos_sim

    def dict_for_art_names(self):
        codex_path = os.path.join(PATH_TO_ROOT, "codexes")
        for cod in os.listdir(codex_path):
            _, art_n = coll.iter_by_docs(cod, codex_path, 'art_name2', 1)
            self.art_names.update(art_n)

    def feature_art_name_intersection(self,  req: str, article: (str, str)) -> int:
        # количество слов из названия статьи, встретившихся в запросе
        return len(set(self.art_names[article].split()) & (set(req.split())))
