from tools.pravoved_recognizer import Request
import joblib
import os
from tools.request_features import create_feature_files_for_all_requests
from tools.relative_paths_to_directories import path_to_directories, CNT_ARTICLES
from xgboost import DMatrix
import pandas as pd
from tools.simple_corp import SimpleCorp
from tools.name_codexes import name_codexes

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25, \
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())

request = "правомерно ли работодатель отказал беременной в переводе на неполный рабочий день?"
new_request = Request(request, "", "").create_dict()

xgb_model = joblib.load("final_xgb_model.sav")

create_feature_files_for_all_requests([new_request], "files/")
features = pd.read_pickle(f"{PATH_TO_FILES}/0.pickle")

x_test = features.drop(['is_rel', '7'], axis=1)
group_test = [CNT_ARTICLES]

test_dmatrix = DMatrix(x_test)
test_dmatrix.set_group(group_test)

pred = xgb_model.predict(test_dmatrix)

corpus = SimpleCorp.load("codexes_corp_articles", os.path.join(PATH_TO_FILES, "corp"))
art_names = SimpleCorp.load('codexes_corp_art_names', f'{PATH_TO_FILES}/corp')

prediction_answer = []
for p, doc_id in zip(pred, list(corpus.corpus.keys())):
    prediction_answer.append((doc_id, p))

prediction_answer.sort(key=lambda x: x[1], reverse=True)

for res in prediction_answer[:5]:
    doc_id = res[0]
    rel = round(res[1], 3)
    print(f"{name_codexes[int(doc_id[0])]}, Cтатья {doc_id[1]}, {art_names.get_doc(doc_id)}, {'%.3f' % rel}")
