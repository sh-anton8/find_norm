from tools.simple_corp import SimpleCorp
from tools.tokenize_docs import Tokenizer
from tools.tfidf import TFIDF
from tools.search import Baseline_Search
import pandas as pd
import os


from tools.relative_paths_to_directories import path_to_directories
PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25, \
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())

# Пример того, как можно предсказать норму по запросу query

# функция релевантности
# features - list признаков для i документа
def relev(features):
    return sum(features) / len(features)


path_to_tf_idf = os.path.join(PATH_TO_TF_IDF, 'tf_idf')
tokenizer = Tokenizer()

# создаем массив поисковиков
searchers_array = []
for i in range(1, 7):
    searchers_array.append(TFIDF.load(path_to_tf_idf + '_' + str(i)))

# поиск по запросу query
b_search = Baseline_Search(tokenizer, searchers_array)
query = 'Симметричные корректировки осуществляются в порядке, установленном настоящей статьей.'
search_result = b_search.search(query, relev)
print(search_result)