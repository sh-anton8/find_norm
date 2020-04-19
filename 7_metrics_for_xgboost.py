from tools import xgboost_metrics
from tools import pravoved_recognizer
import os
from analiz import Analizer
from experiment_analiz import ExpAnalizer

from tools.relative_paths_to_directories import path_to_directories

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25,\
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())

'''
Подсчет метрик ранжирования для предсказанных результатов для тестовой выборки
ВАЖНО! Тестовая и тренировочная выборка такие же как в 6_learning_to_rank.py
'''

TRAIN_SAMPLE = 500 #len(pravoved) = 1429
TEST_SAMPLE = 510


predictions_by_queries = xgboost_metrics.read_predictions_from_file() #чтение предсказаний из файла и их парсинг
pravoved = pravoved_recognizer.norms_codexes_to_normal(os.path.join(PATH_TO_ROOT, "codexes"))
analizer = Analizer(predictions_by_queries, pravoved[TRAIN_SAMPLE: TEST_SAMPLE]) # Класс с метриками
exp_analiz = ExpAnalizer(predictions_by_queries, pravoved[TRAIN_SAMPLE: TEST_SAMPLE], 3, PATH_TO_INV_IND)
analizer.ndcg(30) # подсчет одной из метрик ранжирования (также может быть mrr, map, результирующий график
analizer.map_k(30)
analizer.mrr(30)
#сохранятся files/metrics_count