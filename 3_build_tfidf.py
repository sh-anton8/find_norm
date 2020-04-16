from tools.simple_corp import SimpleCorp
from tools.tokenize_docs import Tokenizer
from tools.tfidf import TFIDF
import typing as tp
import os


from tools.relative_paths_to_directories import path_to_directories
PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25, \
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())

# Построение разных выдов TF-IDF (с разными параматрами)

path_to_tf_idf = os.path.join(PATH_TO_TF_IDF, 'tf_idf')


# функция, строящая словарь параметров для TFIDF.vectorizer
def make_params(ngramm: tp.Tuple[int, int], norm: str = 'l2', use_idf: bool = True, sublinear_tf: bool = False):
    params = {'ngram_range' : ngramm, 'norm' : norm, 'use_idf' : use_idf, 'sublinear_tf' : sublinear_tf}
    return params

# функция, строщая TFIDF по корпусу, токенизатору и параметрам
def build_tf_idfs(corpus, tokenizer, tf_idf_path, num, is_tokenized = True, **params):
    tfidf = TFIDF(tokenizer=tokenizer, vectorizer_params=params)
    tfidf.build_on(corpus, tokenized=is_tokenized)
    tfidf.save(tf_idf_path + '_' + str(num))


# директория на папку с кодексами
codexes_dir = os.path.join(PATH_TO_ROOT, "codexes")

tokenized_corp = SimpleCorp.load('codexes_tokenized_corp_articles', f'{PATH_TO_FILES}\corp')
tokenizer = Tokenizer()

params = make_params((1, 1))
build_tf_idfs(tokenized_corp, tokenizer, path_to_tf_idf, 1, True, **params)
params = make_params((3, 3))
build_tf_idfs(tokenized_corp, tokenizer, path_to_tf_idf, 2, True, **params)
params = make_params((1, 3))
build_tf_idfs(tokenized_corp, tokenizer, path_to_tf_idf, 3, True, **params)
params = make_params((1, 1), norm='l1', use_idf=False)
build_tf_idfs(tokenized_corp, tokenizer, path_to_tf_idf, 4, True, **params)
params = make_params((1, 1), use_idf=False, sublinear_tf=True)
build_tf_idfs(tokenized_corp, tokenizer, path_to_tf_idf, 5, True, **params)
params = make_params((1, 1), sublinear_tf=True)
build_tf_idfs(tokenized_corp, tokenizer, path_to_tf_idf, 6, True, **params)
