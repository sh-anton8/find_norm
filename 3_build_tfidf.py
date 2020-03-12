import os
from tools.tfidf import TFIDF


# Пример того, как можно посчитать и сохранить в файл TFIDF-класс

# директория файла с Обратным инждексом (объектом класса InvIndex)
fnIdxCodex2Article = "idx/codex2article.pickle"

# файл, куда будет сохраняться (или где уже сохранен) объект класса InvIndex
# перед поиском рекомендуется удалять имеющийся файл и считать заново (иначе могут быть проблемы в директориях)
fnTFIDFCodex2Article = "tfidf/tdidf_codex2article.pickle"

# если объект уже создан, то строить нет смысла
# иначе строим и сохраняем объект класса InvIndex по директории fnIdxCodex2Article
if not os.path.isfile(fnTFIDFCodex2Article):
    # первый параметр - директория файла с Обратным инждексом
    # второй параметр - ngramm
    tf = TFIDF(fnIdxCodex2Article, (1, 1))
    tf.count_tf_idf()
    tf.save(fnTFIDFCodex2Article)