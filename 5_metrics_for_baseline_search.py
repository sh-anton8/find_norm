from tools.pravoved_recognizer import Request
from tools.tokenize_docs import Tokenizer
from tools.tfidf import TFIDF
from tools.search import Baseline_Search
from tools.inverse_index import InvIndex
from analiz import Analizer
from tqdm import tqdm
from experiment_analiz import ExpAnalizer

import os
import pickle


from tools.relative_paths_to_directories import path_to_directories
PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25, \
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())

# путь к файлу с массивом ответов
PATH_TO_ANS_FILE = PATH_TO_ROOT + "/files/anwers_baseline"
pravoved = Request.load(os.path.join(PATH_TO_FILES, "pravoved_one_answer.json"))

# проверим файл на существование
# если не существует, то считаем ответы
# иначе переходим к оценке
if not os.path.exists(PATH_TO_ANS_FILE):
    # функция релевантности
    # features - list признаков для i документа
    def average_func(row):
        r_list = list(row)[1:]
        return sum(r_list)/len(r_list)

    path_to_tf_idf = os.path.join(PATH_TO_TF_IDF, 'tf_idf')
    tokenizer = Tokenizer()

    # создаем массив поисковиков
    searchers_array = []
    for i in range(1, 7):
        searchers_array.append(TFIDF.load(path_to_tf_idf + '_' + str(i)))
    searchers_array.append(InvIndex.load(PATH_TO_INV_IND))

    b_search = Baseline_Search(average_func, searchers_array)

    # готовим массив ответов
    ans_arr = [0] * len(pravoved)

    # заполняем массив
    t = tqdm(total=len(pravoved))
    for i in range(len(pravoved)):
        ans_arr[i] = b_search.search(pravoved[i]['question'], len(searchers_array[0].doc_ids), dataFrReturned=False)
        t.update(1)
    t.close()

    # сохраняем в файл
    with open(PATH_TO_ANS_FILE, 'wb') as f:
        pickle.dump(ans_arr, f)

# достаем из файла массив ответов
with open(PATH_TO_ANS_FILE, 'rb') as f:
    arr = pickle.load(f)

# анализируем
analizer = Analizer(arr, pravoved)
analizer2 = ExpAnalizer(arr, pravoved, 3)
#analizer.top_n_cover(21, True)
analizer.map_k(11)
analizer.ndcg(11)
analizer.mrr(11)
analizer2.map_k(11)
analizer2.ndcg(11)
analizer2.mrr(11)

