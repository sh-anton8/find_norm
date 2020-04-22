from tools import xgboost_metrics
from tools.pravoved_recognizer import Request
import os
from analiz import Analizer
from experiment_analiz import ExpAnalizer

from tools.relative_paths_to_directories import path_to_directories

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25,\
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())

'''
Подсчет метрик ранжирования для предсказанных результатов для тестовой выборки
ВАЖНО! Тестовая и тренировочная выборка такие же как в 7_learning_to_rank.py
'''

TRAIN_SAMPLE = (0, 1000)
TEST_SAMPLE = (1001, 1428) #len(pravoved) = 1429



# чтение предсказаний из файла и их парсинг
predictions_by_queries = xgboost_metrics.read_predictions_from_file()

# парсинг правоведа
pravoved = Request.load(os.path.join(PATH_TO_FILES, 'pravoved_one_answer.json'))

# создание класса с метриками
analizer = Analizer(predictions_by_queries, pravoved[TEST_SAMPLE[0]: TEST_SAMPLE[1]])

# создание класса с экспериментальными метриками
exp_analiz = ExpAnalizer(predictions_by_queries, pravoved[TEST_SAMPLE[0]: TEST_SAMPLE[1]], 3)

# подсчет одной из метрик ранжирования (также может быть mrr, map, результирующий график
# сохранятся files/metrics_count

exp_analiz.ndcg(20)
#exp_analiz.mrr(31)
analizer.ndcg(20)