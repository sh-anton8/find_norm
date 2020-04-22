import sklearn
from xgboost import DMatrix
import os
from tools.simple_corp import SimpleCorp
import pandas as pd

from tools.relative_paths_to_directories import path_to_directories, CNT_ARTICLES

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25, \
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())


def predict_xgboost_answers(xgb_model):
    # запись прогноза посчитанной модели на тестовой выборке в виде ((кодекс, статья), вероятность)
    features = pd.read_csv(f"{PATH_TO_LEARNING_TO_RANK}/x_test.csv", sep=',')
    x_test = features.drop(['doc_id', 'is_rel'],axis=1)
    group_test = []
    with open(os.path.join(PATH_TO_LEARNING_TO_RANK, "gr_test.txt"), "r", encoding="utf-8") as f:
        data = f.readlines()
        for line in data:
            group_test.append(int(line.split("\n")[0]))

    test_dmatrix = DMatrix(x_test)
    test_dmatrix.set_group(group_test)

    pred = xgb_model.predict(test_dmatrix)
    corpus = SimpleCorp.load("codexes_corp_articles", os.path.join(PATH_TO_FILES, "corp"))
    prediction_answer = []
    for p, doc_id in zip(pred, list(corpus.corpus.keys()) * (len(pred) // CNT_ARTICLES)):
        prediction_answer.append((doc_id, p))
    predict_file = os.path.join(PATH_TO_LEARNING_TO_RANK, 'prediction_file.txt')
    if os.path.exists(predict_file):
        os.remove(predict_file)
    f = open(predict_file, 'w+', encoding="utf-8")
    predictions = [str(pred) for pred in prediction_answer]
    f.write('\n'.join(predictions))
    f.close()
