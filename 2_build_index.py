from tools.simple_corp import SimpleCorp
from tools.tokenize_docs import Tokenizer
from tools.inverse_index import InvIndex
import os

from tools.relative_paths_to_directories import path_to_directories

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25, \
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())

# Пример того, как можно посчитать и сохранить в файл Обратный индекс

# директория на папку с кодексами
codexes_dir = os.path.join(PATH_TO_ROOT, "codexes")

simple_corp = SimpleCorp.load('codexes_corp_articles', f'{PATH_TO_FILES}/corp')
tokenized_corp = SimpleCorp.load('codexes_tokenized_corp_articles', f'{PATH_TO_FILES}/corp')

tokenizer = Tokenizer()

inv_index = InvIndex(tokenizer=tokenizer)
inv_index.build_on(tokenized_corp, tokenized=True)
inv_index.save(PATH_TO_INV_IND)

query = 'Симметричные корректировки осуществляются в порядке, установленном настоящей статьей.'
search_result = inv_index.search(query)
found_articles = [(simple_corp.get_doc(doc_id), rel) for doc_id, rel in search_result]
print(search_result)
