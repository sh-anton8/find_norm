from tools import xgboost_metrics
from tools import pravoved_recognizer
import os
from analiz import Analizer
from learning_to_rank_06 import TRAIN_SAMPLE, TEST_SAMPLE

from tools.relative_paths_to_directories import path_to_directories

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25,\
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())


predictions_by_queries = xgboost_metrics.read_predictions_from_file()
pravoved = pravoved_recognizer.norms_codexes_to_normal(os.path.join(PATH_TO_ROOT, "codexes"))

analize = Analizer(predictions_by_queries, pravoved[TRAIN_SAMPLE: TEST_SAMPLE])
analize.ndcg(5)