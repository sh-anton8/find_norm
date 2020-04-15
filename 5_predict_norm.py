from tools.simple_corp import SimpleCorp
from tools.tokenize_docs import Tokenizer
from tools.tfidf import TFIDF
from tools.search import Baseline_Search
import typing as tp
import os


from tools.relative_paths_to_directories import path_to_directories
PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25, \
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())

# Пример того, как можно предсказать норму по запросу query

path_to_tf_idf = os.path.join(PATH_TO_TF_IDF, 'tf_idf')

tokenizer = Tokenizer()

simple_corp = SimpleCorp.load('codexes_corp_articles', f'{PATH_TO_FILES}\corp')
tokenized_corp = SimpleCorp.load('codexes_tokenized_corp_articles', f'{PATH_TO_FILES}\corp')

b_search = Baseline_Search(path_to_tf_idf, tokenizer)

query = 'Симметричные корректировки осуществляются в порядке, установленном настоящей статьей.'
search_result = b_search.search(query, threshold=0.1)
found_articles = [(simple_corp.get_doc(doc_id), rel) for doc_id, rel in search_result]
print(search_result)
