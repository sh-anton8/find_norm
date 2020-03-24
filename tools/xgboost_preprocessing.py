import typing as tp
import tools.pravoved_recognizer as prav_rec
import tools.coll as coll
import os
import tools.tokenize_docs as tokenize_docs
from tools import search
from tools import tfidf
from tools import inverse_index as ii
import numpy as np
import features

pravoved_requests = prav_rec.norms_codexes_to_normal("../codexes")


def all_articles_in_codexes(dir: str) -> tp.List[tp.Tuple[str, str]]:
    # Возвращает список всех статей
    all_codex_name = [filename for filename in os.listdir(dir)]
    ans_codexes: tp.List[tp.Tuple[str, str]] = []
    for codex in all_codex_name:
        ans_codexes.extend(coll.iter_by_docs(codex, dir, 'art_name2', 1).keys())
    return ans_codexes


def reqst_features(reqst: prav_rec.Request, path_to_tfidf: str) -> str:
    # Возвращает tfidf запроса
    tfidf_searcher = search.TFIDF_Search(tokenize_docs.Tokenizer('text'), path_to_tfidf)
    return tfidf_searcher.request_tfidf(reqst)


def feature_art_name_intersection(art_name: str, reqst: prav_rec.Request) -> int:
    # количество слов из названия статьи, встретившихся в запросе
    return len(set(list(art_name)) & set(list(reqst.question)))


def inv_ind(dir: str, inv_ind_path: str) -> None:
    # возвращает посчитанный обратный индекс и сохраняет его в файл inv_ind_path
    s = ii.InvIndex(dir, tokenize_docs.Tokenizer('text'))
    s.update_dicts('article')
    s.build_inversed_index('article')
    s.num_tokens_dict_builder()
    s.save(inv_ind_path)


def tfidf_cnt(ngramm: tp.Tuple[int, int], inv_ind_path: str, tfidf_path: str, num:int,
              norm:str='l2', use_idf:bool=True, sublinear_tf:bool=False) -> None:
    # считает tfidf по корпусу с параметрами
    mod = tfidf.TFIDF(inv_ind_path, ngramm=ngramm, norm=norm, use_idf=use_idf, sublinear_tf=sublinear_tf)
    mod.count_tf_idf()
    mod.save(tfidf_path + '_' + str(num))


def cos_simil(reqst: prav_rec.Request, tf_idf_path: str):
    # считает косинусовую сумму
    tfidf_searcher = search.TFIDF_Search(tokenize_docs.Tokenizer('text'), tf_idf_path)
    return tfidf_searcher.cnt_cosine_similarity(reqst.question)


def build_all_indexes_and_tf_idf() -> None:
    # строит все индексы и tf-idf в различных вариациях
    inv_ind("../codexes", "inv_ind")
    tfidf_cnt((1, 1), "inv_ind", "../tf_idf/tf_idf", 1)
    tfidf_cnt((3, 3), "inv_ind", "../tf_idf/tf_idf", 2)
    tfidf_cnt((1, 3), "inv_ind", "../tf_idf/tf_idf", 3)
    tfidf_cnt((1, 1), "inv_ind", "../tf_idf/tf_idf", 4, norm='l1', use_idf=False)
    tfidf_cnt((1, 1), "inv_ind", "../tf_idf/tf_idf", 5, use_idf=False, sublinear_tf=True)
    tfidf_cnt((1, 1), "inv_ind", "../tf_idf/tf_idf", 6, sublinear_tf=True)


def get_features(path_to_tfidf_files: str, reqst: prav_rec.Request) -> tp.List[np.array]:
    # строит косинусовую меру для посчитанных tfidf
    all_tfidf = []
    cos_simil = [None] * 6
    for i in range(6):
        all_tfidf.append(search.TFIDF_Search(tokenize_docs.Tokenizer('text'), os.path.join(path_to_tfidf_files, "tf_idf_{i + 1}")))
    for i, tfidfs in enumerate(all_tfidf):
        cos_simil[i] = tfidfs.cnt_cosine_similarity(reqst.question) #shape (6322, 1)
    return cos_simil

#build_all_indexes_and_tf_idf()
all_features = [[None] * 6322 for i in range(7)]
r = prav_rec.Request('a', 'я мать одиночка и временно не работаю. единственный доход на проживание это пособие с биржи труда. поэтому возникла задолженость', 'a')
inv_index = ii.InvIndex.load("inv_ind")
feature = features.Features("../tools/inv_ind", "../files/my_bm_obj.pickle")
tfidf_file = tfidf.TFIDF.load("../tf_idf/tf_idf_1")
for i in range(len(tfidf_file.num_to_num_dict.keys())):
    all_features[0][i] = feature.get_bm25_feature(r.question, tfidf_file.num_to_num_dict[i])
    all_features[1][i] = feature.get_fmerRelev_feature(r.question, tfidf_file.num_to_num_dict[i])
    all_features[2][i] = feature.get_doc_len_feature(r.question, tfidf_file.num_to_num_dict[i])
print(*all_features[2])

