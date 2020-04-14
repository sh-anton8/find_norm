import tools.tokenize_docs as tokenize_docs
from tools.search import Inv_Ind_Search
from tools.search import TFIDF_Search


# директория на папку с кодексами
fnCollectionDir = "codexes"

# файл, куда будет сохраняться (или где уже сохранен) объект класса InvIndex
# перед поиском рекомендуется удалять имеющийся файл и считать заново (иначе могут быть проблемы в директориях)
fnIdxCodex2Article = "files/inv_ind.pickle"

# файл, куда будет сохраняться (или где уже сохранен) объект класса InvIndex
# перед поиском рекомендуется удалять имеющийся файл и считать заново (иначе могут быть проблемы в директориях)
fnTFIDFCodex2Article = "files/tf_idf/tf_idf_1"

# директория на pravoved
pravovedDir = "files/pravoved_articles.txt"


# Пример поиска основанного на обратном индексе

'''
searcher = Inv_Ind_Search(tokenize_docs.Tokenizer('text'), fnIdxCodex2Article)
searcher.request_processing_pravoved_fmerRelev(fnCollectionDir)
'''

# Пример поиска основанного на TFIDF

'''
tfidf_searcher = TFIDF_Search(tokenize_docs.Tokenizer('text'), fnTFIDFCodex2Article)
tfidf_searcher.request_processing_pravoved(fnCollectionDir, 5)
'''
