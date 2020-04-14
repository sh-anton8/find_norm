import os
from tools.relative_paths_to_directories import path_to_directories

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25,\
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())


def read_predictions_from_file():
    # Чтение предсказаний тестовой выборки из файла
    # Возвращение массива predictions_by_queries
    # predictions_by_queries[j] -- отсортированный по вероятности список статей в нашей нумерации
    # predictions_by_queries[j][k] --  ((кодекс, статья), вероятность того, что jй запрос релевантен k-ой статье
    predictions = []
    with open(os.path.join(PATH_TO_LEARNING_TO_RANK, "prediction_file.txt"), "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.replace('(', '').replace(')', '').replace(' ', ''). replace('\'', '').split('\n')[0].split(',')
            predictions.append(((str(line[0]), str(line[1])), float(line[2])))
    predictions_by_queries = []
    i = 0
    j = 0
    while i + 6322 <= len(predictions):
        predictions_by_queries.append([])
        predictions_by_queries[j].extend(list(sorted(predictions[i:i + 6322], key= lambda x:(x[1], -int(x[0][0])), reverse=True)))
        j += 1
        i += 6322
    return predictions_by_queries
