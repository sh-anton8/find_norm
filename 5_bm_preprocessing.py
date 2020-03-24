from tools.bm25 import My_BM
import os

# директория файла с Обратным инждексом (объектом класса InvIndex)
fnIdxCodex2Article = "idx/codex2article.pickle"

# куда сохраняем
file = "files/my_bm_obj.pickle"

if not os.path.isfile(file):
    obj = My_BM(fnIdxCodex2Article)
    obj.save(file)
