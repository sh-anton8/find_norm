import tools.tokenize_docs as tokenize_docs
from tools.search import Inv_Ind_Search
from tools.search import TFIDF_Search
import os

from tools.relative_paths_to_directories import path_to_directories, CNT_ARTICLES

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25,\
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())

# директория на папку с кодексами
fnCollectionDir = "codexes"

# файл, куда будет сохраняться (или где уже сохранен) объект класса InvIndex
# перед поиском рекомендуется удалять имеющийся файл и считать заново (иначе могут быть проблемы в директориях)
fnIdxCodex2Article = PATH_TO_INV_IND

# файл, куда будет сохраняться (или где уже сохранен) объект класса InvIndex
# перед поиском рекомендуется удалять имеющийся файл и считать заново (иначе могут быть проблемы в директориях)
fnTFIDFCodex2Article = os.path.join(PATH_TO_TF_IDF, "tf_idf")

# директория на pravoved
pravovedDir = os.path.join(PATH_TO_FILES, 'pravoved_articles.txt')


# Пример поиска основанного на обратном индексе


searcher = Inv_Ind_Search(tokenize_docs.Tokenizer('text'), fnIdxCodex2Article)
searcher.request_processing_pravoved_fmerRelev(fnCollectionDir)


# Пример поиска основанного на TFIDF

# 2Anton: при запуске данного куска я не могу запустить ничего, так как путь неправильный!
'''

  File "<ipython-input-10-bb44bb574be8>", line 1, in <module>
    tfidf_searcher = TFIDF_Search(tokenize_docs.Tokenizer('text'), fnTFIDFCodex2Article)

  File "C:\!_Work\НРА\!experiments\ФКН\find_norm\tools\search.py", line 108, in __init__
    self.tfidf_mod = TFIDF.load(tfidf_file)

  File "C:\!_Work\НРА\!experiments\ФКН\find_norm\tools\tfidf.py", line 46, in load
    with open(file, 'rb') as f:

FileNotFoundError: [Errno 2] No such file or directory: 'C:\\!_Work\\НРА\\!experiments\\ФКН\\find_norm\\files\\tf_idf\\tf_idf'
tfidf_searcher = TFIDF_Search(tokenize_docs.Tokenizer('text'), fnTFIDFCodex2Article)
tfidf_searcher.request_processing_pravoved(fnCollectionDir, 5)
'''
