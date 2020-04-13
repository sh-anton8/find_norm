import os
from tools.tfidf import TFIDF


from tools.relative_paths_to_directories import path_to_directories
PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25, \
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())
# Пример того, как можно посчитать и сохранить в файл TFIDF-класс

# директория файла с Обратным инждексом (объектом класса InvIndex)
fnIdxCodex2Article = PATH_TO_INV_IND

# файл, куда будет сохраняться (или где уже сохранен) объект класса InvIndex
# перед поиском рекомендуется удалять имеющийся файл и считать заново (иначе могут быть проблемы в директориях)
fnTFIDFCodex2Article = os.path.join(PATH_TO_TF_IDF, 'tf_idf')

# если объект уже создан, то строить нет смысла
# иначе строим и сохраняем объект класса InvIndex по директории fnIdxCodex2Article
if not os.path.isfile(fnTFIDFCodex2Article):
    # первый параметр - директория файла с Обратным инждексом
    # второй параметр - ngramm
    tf = TFIDF(fnIdxCodex2Article, (1, 1))
    tf.count_tf_idf()
    tf.save(fnTFIDFCodex2Article)
