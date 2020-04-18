from tools.tokenize_docs import Tokenizer
from tools import coll
from tools.simple_corp import SimpleCorp
import os
from tqdm import tqdm

from tools.relative_paths_to_directories import path_to_directories

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25, \
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())

# директория на папку с кодексами
codexes_dir = os.path.join(PATH_TO_ROOT, "codexes")

tokenizer = Tokenizer()
simple_corp = SimpleCorp()

for filename in tqdm(os.listdir(codexes_dir)):
    d1, _ = coll.iter_by_docs(filename, codexes_dir, 'article', 0)
    for doc_id, doc_text in d1.items():
        simple_corp.add_doc(doc_id, doc_text)

tokenized_corp = SimpleCorp()
tokenized_corp.make_from(simple_corp, tokenizer)

simple_corp.save('codexes_corp_articles', os.path.join(PATH_TO_FILES, "corp"))
tokenized_corp.save('codexes_tokenized_corp_articles', os.path.join(PATH_TO_FILES, "corp"))
