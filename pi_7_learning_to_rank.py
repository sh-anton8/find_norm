from tools import xgboost_model, xgboost_predict, request_features, xgboost_features
import os
import joblib

from tools.relative_paths_to_directories import path_to_directories

'''
Выбор тренировочной и тестовой выборки 
Тренировочная выборка pravoved[:TRAIN_SAMPLE]
Тестовая: pravoved[TRAIN_SAMPLE:TEST_SAMPLE]
ВАЖНО!
Тренировочная и тестовая выборки в pi_7_learning_to_rank.py и pi_8_metrics_for_xgboost.py
должны быть одинаковые
'''
TRAIN_SAMPLE = (428, 1429)
TEST_SAMPLE = (0, 428)

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25,\
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())

xgboost_features.pravoved_features_to_files(TRAIN_SAMPLE, TEST_SAMPLE) # посчитанные признаки записываются в папку files/learning_to_rank
xgb_model = xgboost_model.train_xgboost_model() # строится модель на основе выделенных признаков
joblib.dump(xgb_model, "final_xgb_model.sav")
xgboost_predict.predict_xgboost_answers(xgb_model) # считаются предсказания для тестовой выборки и записываются в files/learning_to_rank
