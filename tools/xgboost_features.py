import os
from tools.pravoved_recognizer import Request
from tqdm import tqdm
import pandas as pd

from tools.relative_paths_to_directories import path_to_directories, CNT_ARTICLES

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25,\
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())

PATH_TO_FEATURE_DF = os.path.join(PATH_TO_FILES, 'features_df')

pravoved_requests = Request.load(os.path.join(PATH_TO_FILES, 'pravoved_one_answer.json'))


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

    feature_file_train = os.path.join(PATH_TO_LEARNING_TO_RANK, 'x_train.csv')
    feature_file_test = os.path.join(PATH_TO_LEARNING_TO_RANK, 'x_test.csv')

    train_features = []
    # запись признаков тренировочной выборки
    t = tqdm(total=len(train_pravoved_requests))
    for i, req in enumerate(train_pravoved_requests):
        path_to_file = f'{PATH_TO_FEATURE_DF}/{i}.pickle'
        train_features.append(pd.read_pickle(path_to_file))
        t.update(1)
    t.close()
    res = pd.concat(train_features)
    res.to_csv(feature_file_train)

    # запись признаков тестовой выборки
    test_features = []
    t = tqdm(total=len(test_pravoved_requests))
    for i, req in enumerate(test_pravoved_requests):
        path_to_file = f'{PATH_TO_FEATURE_DF}/{i + test_sample[0]}.pickle'
        test_features.append(pd.read_pickle(path_to_file))
        t.update(1)
    t.close()
    res = pd.concat(test_features)
    res.to_csv(feature_file_test)
