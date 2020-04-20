from tools.request_features import create_feature_files_for_all_requests
from tools.pravoved_recognizer import Request
import os

from tools.relative_paths_to_directories import path_to_directories

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25, \
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())

# создание таблиц с признаками для каждого запроса
# если таблицы уже созданы, этот шаг можно пропустить

# чтение запросов правоведа
pravoved_requests = Request.load(os.path.join(PATH_TO_FILES, 'pravoved_one_answer.json'))

# подсчет признаков для всех запросов в pravoved_requests
create_feature_files_for_all_requests(pravoved_requests, os.path.join(PATH_TO_FILES, 'features_df'))