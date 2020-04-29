import typing as tp
import os
from tqdm import tqdm
from tools.simple_corp import SimpleCorp
from typing import List
import pandas as pd
from tools.inverse_index import InvIndex
from tools.tfidf import TFIDF

from tools.relative_paths_to_directories import path_to_directories

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25,\
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())

CORPUS = SimpleCorp.load("codexes_corp_articles", os.path.join(PATH_TO_FILES, "corp"))

def is_article_relev(r, art: tp.Tuple[str, str]) -> tp.Tuple[tp.Tuple[str, str], int]:
    # релевантен ли запрос статье
    if (str(r['codex']), str(r['norm'])) == art:
        return art, 1
    return art, 0


def relev_articles_for_features(request):
    # возвращение вектора релевантностей для корпуса статей по данному запросу
    relevs = []
    for i, doc_id in enumerate(CORPUS.corpus.keys()):
        relevs.append(is_article_relev(request, doc_id))
    return relevs


def find_list_of_features_for_request():
    # подсчет всех признаков
    searchers_array = []
    for i in range(1, 7):
        searchers_array.append(TFIDF.load(f'{PATH_TO_TF_IDF}/tf_idf_{i}'))
    searchers_array.append(InvIndex.load(PATH_TO_INV_IND))
    return searchers_array


def request_to_pandas(features: List[TFIDF], isrelev: tp.List[tp.Tuple[tp.Tuple[str, str], int]],
                      query: str):
    # создание таблицы, где в столбцах будут будут значения признаков,
    # по строками статьи, которым эти признаки соответствуют
    result = pd.DataFrame(isrelev, columns=['doc_id', 'is_rel'])
    for i, feature in enumerate(features):
        sim = feature.search(query, len(features[0].doc_ids), 0)
        newDF = pd.DataFrame(sim, columns=['doc_id', str(i + 1)]).set_index('doc_id')
        result = result.join(newDF, on='doc_id')
    result.set_index('doc_id', inplace=True)
    return result


def save_feature_table(df: pd.DataFrame, path: str):
    # сохранение таблицы признаков в файл
    df.to_pickle(path)


def create_feature_file_for_request(request, path_to_file, all_features) -> None:
    # список релевантностей для статей корпуса
    relevs = relev_articles_for_features(request)

    # создание списка признаков для запроса по корпус

    # создание таблицы признаков для request
    table_features = request_to_pandas(all_features, relevs, request['theme'])

    # сохранение таблицы в директорию
    save_feature_table(table_features, path_to_file)


def create_feature_files_for_all_requests(requests: tp.List[tp.Dict[str, str]], path_to_dir: str) -> None:
    # запись в директорию файлов с таблицами признаков для каждого запроса

    # класс со всеми признаками и подсчет всех tf-idf

    # создание списка признаков для запроса по корпусу
    all_features = find_list_of_features_for_request()

    # итерация по всем запросам
    t = tqdm(total=len(requests))
    for i, request in enumerate(requests):
        create_feature_file_for_request(request, os.path.join(path_to_dir, f'{i}.pickle'), all_features)
        t.update(1)
    t.close()
