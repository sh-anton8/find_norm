import sklearn
import xgboost
import os
from tools.relative_paths_to_directories import path_to_directories
from xgboost import XGBRanker

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25,\
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())


def train_xgboost_model():
    x_train, y_train = sklearn.datasets.load_svmlight_file(os.path.join(PATH_TO_LEARNING_TO_RANK, 'x_train.txt'))
    train_dmatrix = xgboost.DMatrix(x_train, y_train)
    group_train = []
    group_test = []
    with open(os.path.join(PATH_TO_LEARNING_TO_RANK, "gr_train.txt"), "r") as f:
        data = f.readlines()
        for line in data:
            group_train.append(int(line.split("\n")[0]))

    with open(os.path.join(PATH_TO_LEARNING_TO_RANK, "gr_test.txt"), "r") as f:
        data = f.readlines()
        for line in data:
            group_test.append(int(line.split("\n")[0]))

    params = {'objective': 'rank:ndcg', 'learning_rate': 0.1,
              'gamma': 1.0, 'min_child_weight': 0.1,
              'max_depth': 6, 'n_estimators': 4}
    model = xgboost.sklearn.XGBRanker(**params)
    model.fit(x_train, y_train, group_train, verbose=True)
    return model
