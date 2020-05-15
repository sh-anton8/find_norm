from tools.simple_corp import SimpleCorp
from tools.tokenize_docs import Tokenizer
from tools.tfidf import TFIDF
from tools.inverse_index import InvIndex
from tools.search import Baseline_Search
from tools.name_codexes import name_codexes

import os


from tools.relative_paths_to_directories import path_to_directories
PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25, \
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())

art_names = SimpleCorp.load("codexes_corp_art_names", f"{PATH_TO_FILES}/corp")

# Пример того, как можно предсказать норму по запросу query

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

# поиск по запросу query
b_search = Baseline_Search(average_func, searchers_array)
query = 'Симметричные корректировки осуществляются в порядке, установленном настоящей статьей.'
search_result = b_search.search(query, topN=10, dataFrReturned=True)
for index, row in search_result.iterrows():
    doc_id = row['doc_id']
    print(f"{name_codexes[int(doc_id[0])]}, Cтатья {doc_id[1]}, {art_names.get_doc(row['doc_id'])}, {round(row['FinalRel'], 2)}")
