from tools.bm25 import My_BM
import os

from tools.relative_paths_to_directories import path_to_directories

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25,\
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())


# директория файла с Обратным инждексом (объектом класса InvIndex)
fnIdxCodex2Article = PATH_TO_INV_IND

# куда сохраняем
file = PATH_TO_BM_25

if not os.path.isfile(file):
    obj = My_BM(fnIdxCodex2Article)
    obj.save(file)
