import tools.tokenize_docs as tokenize_docs
from tools.inverse_index import InvIndex
import os


from tools.relative_paths_to_directories import path_to_directories

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25, \
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())

# Пример того, как можно посчитать и сохранить в файл Обратный индекс

# директория на папку с кодексами
fnCollectionDir = os.path.join(PATH_TO_ROOT, "codexes")

# файл, куда будет сохраняться (или где уже сохранен) объект класса InvIndex
# перед поиском рекомендуется удалять имеющийся файл и считать заново (иначе могут быть проблемы в директориях)
fnIdxCodex2Article = PATH_TO_INV_IND

# если объект уже создан, то строить нет смысла
# иначе строим и сохраняем объект класса InvIndex по директории fnIdxCodex2Article
if not os.path.isfile(fnIdxCodex2Article):
    s = InvIndex(fnCollectionDir, tokenize_docs.Tokenizer(''))
    s.update_dicts('article')
    s.build_inversed_index('article')
    s.num_tokens_dict_builder()
    s.save(fnIdxCodex2Article)
