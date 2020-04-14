import sklearn
from xgboost import DMatrix
import os
from tools.tfidf import TFIDF

from tools.relative_paths_to_directories import path_to_directories, CNT_ARTICLES

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25, \
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())


def predict_xgboost_answers(xgb_model):
    # запись прогноза посчитанной модели на тестовой выборке в виде ((кодекс, статья), вероятность)
    load_tfidf_1 = TFIDF.load(os.path.join(PATH_TO_TF_IDF, 'tf_idf_1'))
    x_test, y_test = sklearn.datasets.load_svmlight_file(os.path.join(PATH_TO_LEARNING_TO_RANK, 'x_test.txt'))
    test_dmatrix = DMatrix(x_test)
    pred = xgb_model.predict(x_test)
    prediction_answer = []
    for i, p in enumerate(pred):
        prediction_answer.append((load_tfidf_1.num_to_num_dict[i % CNT_ARTICLES], p))
    predict_file = os.path.join(PATH_TO_LEARNING_TO_RANK, 'prediction_file.txt')
    if os.path.exists(predict_file):
        os.remove(predict_file)
    f = open(predict_file, 'w+', encoding="utf-8")
    predictions = [str(pred) for pred in prediction_answer]
    f.write('\n'.join(predictions))
    f.close()
