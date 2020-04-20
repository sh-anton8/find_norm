import os
from tools.pravoved_recognizer import Request
from tqdm import tqdm
import pandas as pd

from tools.relative_paths_to_directories import path_to_directories, CNT_ARTICLES

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25,\
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())

PATH_TO_FEATURE_DF = os.path.join(PATH_TO_FILES, 'features_df')

pravoved_requests = Request.load(os.path.join(PATH_TO_FILES, 'pravoved_one_answer.json'))


def preparation_features_for_xgboost(num: int, ff) -> None:
    # запись таблицы признаков с номером num в file_to_save
    path_to_file = os.path.join(PATH_TO_FEATURE_DF, f'{num}.pickle')
    feature_df = pd.read_pickle(path_to_file)
    feature_df.to_string()
    '''
    for index, row in feature_df.iterrows():
        feature_list = [str(row['is_relev'])]
        for c in feature_df.columns:
            if c != 'is_relev' and c != 'doc_id':
                feature_list.append(f'{int(c) + 1}:{row[c]}')
        ff.write(' '.join(feature_list) + '\n')
    '''


def create_group_file(requests_list: str, path_to_file: str) -> None:
    # создание group file (распределение статей по запросам)
    with open(path_to_file, 'w+', encoding='utf-8') as f:
        for i in range(len(requests_list)):
            f.write(f'{CNT_ARTICLES}\n')


def delete_if_exist(path_to_file: str) -> None:
    # удаление файла, если он существует
    if os.path.exists(path_to_file):
        os.remove(path_to_file)


def pravoved_features_to_files(train_sample: (int, int), test_sample: (int, int)) -> None:
    # создание тестовой и тренировочной выборки
    # Для каждого запроса записываются его признаки в файлы

    # удаление файлов с разделением на группы и
    # файлов с признаками для тренирочной и тестовой выборками
    delete_if_exist(os.path.join(PATH_TO_LEARNING_TO_RANK, 'x_train.txt'))
    delete_if_exist(os.path.join(PATH_TO_LEARNING_TO_RANK, 'x_test.txt'))
    delete_if_exist(os.path.join(PATH_TO_LEARNING_TO_RANK, 'gr_train.txt'))
    delete_if_exist(os.path.join(PATH_TO_LEARNING_TO_RANK, 'gr_test.txt'))

    # создание тренировочной выборки
    train_pravoved_requests = pravoved_requests[train_sample[0]:train_sample[1] + 1]

    # создание тестовой выборки
    test_pravoved_requests = pravoved_requests[test_sample[0]:test_sample[1] + 1]

    # создание групп для тестовой и тренировочной выборок
    create_group_file(train_pravoved_requests, os.path.join(PATH_TO_LEARNING_TO_RANK, "gr_train.txt"))
    create_group_file(test_pravoved_requests, os.path.join(PATH_TO_LEARNING_TO_RANK, "gr_test.txt"))

    feature_file_train = os.path.join(PATH_TO_LEARNING_TO_RANK, 'x_train.txt')
    feature_file_test = os.path.join(PATH_TO_LEARNING_TO_RANK, 'x_test.txt')

    # запись признаков тренировочной выборки
    with open(feature_file_train, 'a+') as ftrain:
        t = tqdm(total=len(train_pravoved_requests))
        for i, req in enumerate(train_pravoved_requests):
            preparation_features_for_xgboost(train_sample[0] + i, ftrain)
            t.update(1)
        t.close()

    # запись признаков тестовой выборки
    with open(feature_file_test, 'a+') as ftest:
        t = tqdm(total=len(test_pravoved_requests))
        for i, req in enumerate(test_pravoved_requests):
            preparation_features_for_xgboost(test_sample[0] + i, ftest)
            t.update(1)
        t.close()
