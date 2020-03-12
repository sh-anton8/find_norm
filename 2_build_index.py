import tools.tokenize_docs as tokenize_docs
from tools.inverse_index import InvIndex
import os

# Пример того, как можно посчитать и сохранить в файл Обратный индекс

# директория на папку с кодексами
fnCollectionDir = "codexes"

# файл, куда будет сохраняться (или где уже сохранен) объект класса InvIndex
# перед поиском рекомендуется удалять имеющийся файл и считать заново (иначе могут быть проблемы в директориях)
fnIdxCodex2Article = "idx/codex2article.pickle"

# если объект уже создан, то строить нет смысла
# иначе строим и сохраняем объект класса InvIndex по директории fnIdxCodex2Article
if not os.path.isfile(fnIdxCodex2Article):
    s = InvIndex(fnCollectionDir, tokenize_docs.Tokenizer(''))
    s.update_dicts('article')
    s.build_inversed_index('article')
    s.num_tokens_dict_builder()
    s.save(fnIdxCodex2Article)