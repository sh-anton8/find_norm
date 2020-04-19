from tools.pravoved_recognizer import Request
import typing as tp
import os
from tools.features import Features
from tools import pravoved_recognizer
from tqdm import tqdm
from tools.simple_corp import SimpleCorp
from typing import List
import pandas as pd
from collections import defaultdict

from tools.relative_paths_to_directories import path_to_directories, CNT_ARTICLES

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25,\
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())

CORPUS = SimpleCorp.load("codexes_corp_articles", os.path.join(PATH_TO_FILES, "corp"))

FEATURES_NUM = 6


def is_article_relev(r: Request, art: tp.Tuple[str, str]) -> bool:
    # релевантен ли запрос статье
    if (str(r.codex), str(r.norm)) == art:
        return True
    return False


def request_to_pandas(features: List[List[float]]):
    d = defaultdict(list)
    feature_name = 'Relev '
    for i, doc_id in enumerate(CORPUS.corpus.keys()):
        d['doc_id'].append(doc_id)
        for j, f in enumerate(features):
            d[feature_name + str(j)].append(f[i])
    df = pd.DataFrame(data=d, index=d['doc_id'])
    df.set_index('doc_id', inplace=True)
    return df


def request_feature_to_file(request: Request, path_to_featute_file: str,
                               feature: Features, is_train=False):
    #записывает целевую переменную и признаки данного запроса в файл
    with open(path_to_featute_file, 'a+', encoding='utf-8') as x:
        all_features_for_request = [[0] * CNT_ARTICLES for _ in range(FEATURES_NUM)]
        cos_simils = feature.features_cos_sim(request)
        for i, cs in enumerate(cos_simils):
            all_features_for_request[i] = cs
        for i, doc_id in enumerate(CORPUS.corpus.keys()):
            is_relev = is_article_relev(request, doc_id)
            if is_train:
                x.write(f'{int(is_relev)} ')
            for j in range(FEATURES_NUM):
                x.write(f'{j + 1}:{all_features_for_request[j][i]}')
                if j != FEATURES_NUM - 1:
                    x.write(' ')
            x.write('\n')


def find_list_of_features_for_request(request: Request, feature: Features):
    all_features_for_request = [[0] * CNT_ARTICLES for _ in range(FEATURES_NUM)]
    cos_simils = feature.features_cos_sim(request)
    for i, cs in enumerate(cos_simils):
        all_features_for_request[i] = cs
    return all_features_for_request


def create_group_file(requests_list: str, path_to_file: str) -> None:
    # создает group file (распределение статей по запросам)
    with open(path_to_file, 'w+', encoding='utf-8') as f:
        for i in range(len(requests_list)):
            f.write(f'{CNT_ARTICLES}\n')


def delete_if_exist(path_to_file: str) -> None:
    # удаление файла, если он существует
    if os.path.exists(path_to_file):
        os.remove(path_to_file)


def pravoved_features_to_files(train_sample: int, test_sample: int) -> None:
    # Создаются тестовая и тренировочная выборки
    # Для каждого запроса записываются его признаки в файлы

    delete_if_exist(os.path.join(PATH_TO_LEARNING_TO_RANK, 'x_train.txt'))
    delete_if_exist(os.path.join(PATH_TO_LEARNING_TO_RANK, 'x_test.txt'))
    delete_if_exist(os.path.join(PATH_TO_LEARNING_TO_RANK, 'gr_train.txt'))
    delete_if_exist(os.path.join(PATH_TO_LEARNING_TO_RANK, 'gr_test.txt'))

    pravoved_requests = pravoved_recognizer.norms_codexes_to_normal(os.path.join(PATH_TO_ROOT, "codexes"))

    train_pravoved_requests = pravoved_requests[:train_sample]
    test_pravoved_requests = pravoved_requests[train_sample:test_sample]

    create_group_file(train_pravoved_requests, os.path.join(PATH_TO_LEARNING_TO_RANK, "gr_train.txt"))
    create_group_file(test_pravoved_requests, os.path.join(PATH_TO_LEARNING_TO_RANK, "gr_test.txt"))

    feature = Features(PATH_TO_INV_IND, os.path.join(PATH_TO_FILES, "bm_25.pickle"))
    feature.load_all_tfidf()

    t = tqdm(total=len(train_pravoved_requests))
    for i, req in enumerate(train_pravoved_requests):
        request_feature_to_file(req, os.path.join(PATH_TO_LEARNING_TO_RANK, "x_train.txt"), feature, is_train=True)
        t.update(1)
    t.close()

    t = tqdm(total=len(test_pravoved_requests))
    for i, req in enumerate(test_pravoved_requests):
        request_feature_to_file(req, os.path.join(PATH_TO_LEARNING_TO_RANK, "x_test.txt"), feature, is_train=True)
        t.update(1)
    t.close()
