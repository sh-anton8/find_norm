from tools.pravoved_recognizer import Request
import typing as tp
import os
from tools.features import Features
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


def is_article_relev(r, art: tp.Tuple[str, str]) -> bool:
    # релевантен ли запрос статье
    if (str(r['codex']), str(['r.norm'])) == art:
        return True
    return False


def relev_articles_for_features(request):
    # возвращение вектора релевантностей для корпуса статей по данному запросу
    relevs = []
    for i, doc_id in enumerate(CORPUS.corpus.keys()):
        is_relev = is_article_relev(request, doc_id)
        relevs.append(int(is_relev))
    return relevs


def find_list_of_features_for_request(request: Request, feature: Features):
    # подсчет всех признаков
    all_features_for_request = [[0] * CNT_ARTICLES for _ in range(FEATURES_NUM)]
    cos_simils = feature.features_cos_sim(request)
    for i, cs in enumerate(cos_simils):
        all_features_for_request[i] = cs
    return all_features_for_request


def request_to_pandas(features: List[List[float]], isrelev: List[int]):
    # создание таблицы, где в столбцах будут будут значения признаков,
    # по строками статьи, которым эти признаки соответствуют
    d = defaultdict(list)
    for i, doc_id in enumerate(CORPUS.corpus.keys()):
        d['doc_id'].append(doc_id)
        for j, f in enumerate(features):
            d[str(j)].append(f[i])
    d['is_relev'] = isrelev
    df = pd.DataFrame(data=d, index=d['doc_id'])
    df.set_index('doc_id', inplace=True)
    return df


def save_feature_table(df: pd.DataFrame, path: str):
    # сохранение таблицы признаков в файл
    df.to_pickle(path)


def create_feature_file_for_request(request, feature: Features, path_to_file) -> None:
    # список релевантностей для статей корпуса
    relevs = relev_articles_for_features(request)

    # создание списка признаков для запроса по корпусу
    all_features = find_list_of_features_for_request(request, feature)

    # создание таблицы признаков для request
    table_features = request_to_pandas(all_features, relevs)

    # сохранение таблицы в директорию
    save_feature_table(table_features, path_to_file)


def create_feature_files_for_all_requests(requests: tp.List[Request], path_to_dir: str) -> None:
    # запись в директорию файлов с таблицами признаков для каждого запроса

    # класс со всеми признаками и подсчет всех tf-idf
    feature = Features(PATH_TO_INV_IND, os.path.join(PATH_TO_FILES, "bm_25.pickle"))
    feature.load_all_tfidf()

    # итерация по всем запросам
    t = tqdm(total=len(requests))
    for i, request in enumerate(requests):
        create_feature_file_for_request(request, feature, os.path.join(path_to_dir, f'{i}.pickle'))
        t.update(1)
    t.close()
