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
    group_test = []
    with open(os.path.join(PATH_TO_LEARNING_TO_RANK, "gr_test.txt"), "r", encoding="utf-8") as f:
        data = f.readlines()
        for line in data:
            group_test.append(int(line.split("\n")[0]))
    test_dmatrix = DMatrix(x_test)
    test_dmatrix.set_group(group_test)
    pred = xgb_model.predict(test_dmatrix)
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
