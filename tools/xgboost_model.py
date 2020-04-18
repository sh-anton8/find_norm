import sklearn
import xgboost as xgb
import os
from tools.relative_paths_to_directories import path_to_directories

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25,\
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())


def train_xgboost_model():
    # построение xgbRank модели и ее возвращение
    # Парсинг записанных признаков и групп и обучение модели с их помощью
    x_train, y_train = sklearn.datasets.load_svmlight_file(os.path.join(PATH_TO_LEARNING_TO_RANK, 'x_train.txt'))

    group_train = []
    with open(os.path.join(PATH_TO_LEARNING_TO_RANK, "gr_train.txt"), "r", encoding="utf-8") as f:
        data = f.readlines()
        for line in data:
            group_train.append(int(line.split("\n")[0]))
    train_dmatrix = xgb.DMatrix(x_train, y_train)
    train_dmatrix.set_group(group_train)

    params = {'objective': 'rank:ndcg', 'eta': 0.1, 'gamma': 1.0,
              'min_child_weight': 0.1, 'max_depth': 6}
    xgb_model = xgb.train(params, train_dmatrix, num_boost_round=4)
    return xgb_model
