import os

CNT_ARTICLES = 6322

#Рассчитываются пути до основых файлов исходя из текущей директории

def path_to_directories(path_to_cur_dir):
    if path_to_cur_dir.endswith('tools'):
        PATH_TO_ROOT = os.path.join(path_to_cur_dir, '..')
        PATH_TO_TOOLS = path_to_cur_dir
        PATH_TO_FILES = os.path.join(path_to_cur_dir, '..', 'files')
    else:
        PATH_TO_TOOLS = os.path.join(path_to_cur_dir, 'tools')
        PATH_TO_ROOT = path_to_cur_dir
        PATH_TO_FILES = os.path.join(path_to_cur_dir, 'files')
    PATH_TO_INV_IND = os.path.join(PATH_TO_FILES, 'inv_ind.pickle')
    PATH_TO_BM_25 = os.path.join(PATH_TO_FILES, 'bm_25.pickle')
    PATH_TO_TF_IDF = os.path.join(PATH_TO_FILES, 'tf_idf')
    PATH_TO_LEARNING_TO_RANK = os.path.join(PATH_TO_FILES, 'learning_to_rank')
    return PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25, PATH_TO_LEARNING_TO_RANK

