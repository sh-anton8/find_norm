from tools import xgboost_model, xgboost_predict, xgboost_features, xgboost_metrics
import os

from tools.relative_paths_to_directories import path_to_directories

TRAIN_SAMPLE = 10
TEST_SAMPLE = 13

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25,\
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())


xgboost_features.features_to_files(TRAIN_SAMPLE, TEST_SAMPLE)
xgb_model = xgboost_model.train_xgboost_model()
xgboost_predict.predict_xgboost_answers(xgb_model)
