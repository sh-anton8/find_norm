from tools.simple_corp import SimpleCorp
from tools.tokenize_docs import Tokenizer
from tools.inverse_index import InvIndex
from tools.name_codexes import name_codexes
import os

from tools.relative_paths_to_directories import path_to_directories

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25, \
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())

# Пример того, как можно посчитать и сохранить в файл Обратный индекс

# директория на папку с кодексами

def predict_norm(query):
    tokenized_corp = SimpleCorp.load('codexes_tokenized_corp_articles', f'{PATH_TO_FILES}/corp')
    art_names = SimpleCorp.load('codexes_corp_art_names', f'{PATH_TO_FILES}/corp')

    tokenizer = Tokenizer()

    inv_index = InvIndex(tokenizer=tokenizer)
    inv_index.build_on(tokenized_corp, tokenized=True)
    inv_index.save(PATH_TO_INV_IND)

    search_result = inv_index.search(query)
    valid_answer = []
    for res in search_result[:5]:
        doc_id = res[0]
        print(f"{name_codexes[int(doc_id[0])]}, Cтатья {doc_id[1]}, {art_names.get_doc(doc_id)}, {round(res[1], 2)}")
        valid_answer.append(f"{name_codexes[int(doc_id[0])]}, Cтатья {doc_id[1]}, {art_names.get_doc(doc_id)}")
    return valid_answer
