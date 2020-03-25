import numpy as np
import typing as tp
import os
import random
import xgboost as xgb
from xgboost import DMatrix
import sklearn
import tools.pravoved_recognizer as prav_rec
import tools.coll as coll
import tools.tokenize_docs as tokenize_docs
from tools import search
from tools import tfidf
from tools import inverse_index as ii
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


def cos_simil(reqst: prav_rec.Request, tfidf_searcher: search.TFIDF_Search):
    # считает косинусовую меру
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


def is_article_relev(r: prav_rec.Request, art: tp.Tuple[str, str]) -> bool:
    if (str(r.codex), str(r.norm)) == art:
        return True
    return False


def get_features(path_to_tfidf_files) -> tp.List[np.array]:
    # строит косинусовую меру для посчитанных tfidf
    all_tfidf = []
    for i in range(6):
        all_tfidf.append(search.TFIDF_Search(tokenize_docs.Tokenizer('text'), os.path.join(path_to_tfidf_files, f"tf_idf_{i + 1}")))
    return all_tfidf


all_tfidf = get_features("../tf_idf")
feature = features.Features("../tools/inv_ind", "../files/my_bm_obj.pickle")
tfidf_file = tfidf.TFIDF.load("../tf_idf/tf_idf_1")


def get_features(reqst: prav_rec.Request) -> tp.List[np.array]:
    cos_simil = [None] * 6
    for i, tfidfs in enumerate(all_tfidf):
        cos_simil[i] = tfidfs.cnt_cosine_similarity(reqst.question) #shape (6322, 1)
    return cos_simil

#build_all_indexes_and_tf_idf()


def find_feautures_for_request(req_num: int, request: prav_rec.Request, path_to_featute_file: str,
                               is_train=False):
    #записывает целевую переменную и признаки данного признака в файл
    with open(path_to_featute_file, 'a+') as x:
        all_features_for_request = [[0] * 6322 for _ in range(9)]
        for i in range(len(tfidf_file.num_to_num_dict.keys())):
            all_features_for_request[0][i] = feature.get_bm25_feature(request.question, tfidf_file.num_to_num_dict[i])
            #all_features_for_request[1][i] = feature.get_fmerRelev_feature(request.question, tfidf_file.num_to_num_dict[i])
            all_features_for_request[2][i] = feature.get_doc_len_feature(request.question, tfidf_file.num_to_num_dict[i])
        cos_simils = get_features(request)
        for i, cs in enumerate(cos_simils):
            all_features_for_request[i + 3] = [el[0] for el in cs]
        for i in range(6322):
            is_relev = is_article_relev(request, tfidf_file.num_to_num_dict[i])
            if is_train:
                if is_relev:
                    x.write('1 ')
                else:
                    x.write('0 ')
            #x.write(f'qid: {req_num + 1} ')
            for j in range(9):
                x.write(f'{j + 1}:{all_features_for_request[j][i]}')
                if j != 8:
                    x.write(' ')
            x.write('\n')


def create_group_file(requests_list: str, path_to_file: str) -> None:
    # создает group file (распределение статей по запросам)
    with open(path_to_file, 'w+') as f:
        for i in range(len(requests_list)):
            f.write('6322\n')

if os.path.exists('req_x.txt'):
    os.remove('req_x.txt')
if os.path.exists('req_y.txt'):
    os.remove('req_y.txt')
if os.path.exists('req_x_test.txt'):
    os.remove('req_x_test.txt')


random.shuffle(pravoved_requests)
train_pravoved_requests = pravoved_requests[:15]
test_pravoved_requests = pravoved_requests[7:8]

for i, req in enumerate(train_pravoved_requests):
    find_feautures_for_request(i, req, "req_x.txt", is_train=True)

for i, req in enumerate(test_pravoved_requests):
    find_feautures_for_request(i, req, "req_x_test.txt", is_train=True)

x_train, y_train = sklearn.datasets.load_svmlight_file('req_x.txt')
x_test, y_test = sklearn.datasets.load_svmlight_file('req_x_test.txt')
train_dmatrix = DMatrix(x_train, y_train)
test_dmatrix = DMatrix(x_test)
create_group_file(train_pravoved_requests, "gr_train.txt")
create_group_file(test_pravoved_requests, "gr_test.txt")
group_train = []
with open("gr_train.txt", "r") as f:
    data = f.readlines()
    for line in data:
        group_train.append(int(line.split("\n")[0]))

train_dmatrix.set_group(group_train)
print(group_train)
params = {'objective': 'rank:ndcg', 'eta': 0.1, 'gamma': 1.0,
          'min_child_weight': 0.1, 'max_depth': 6}
xgb_model = xgb.train(params, train_dmatrix, num_boost_round=4)
pred = xgb_model.predict(test_dmatrix)
print(pred.shape)


