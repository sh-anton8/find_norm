# Антон

#coding: utf-8

import pandas as pd
import os


# базовый поисковик (агрегатор)
# searchers - массив поисковиков (например массив классов TFIDF)
class Baseline_Search:
    def __init__(self, relev_func=None, searchers=None):
        self.relev_func = relev_func
        self.searchers = searchers
        self.df_to_pickle = None


    # функция поиска (запрос, # документов в топе,
    # вернуть в виде DataFrame (по умолчанию) или массив ответов (для анализа))
    # функция релевантонсти принимает массив признаков для i документа и возвращает его релевантность
    def search(self, query, topN = 10, dataFrReturned = True):

        # проверка, что передан зотя бы 1 поисковик
        if not self.searchers:
            print("ERROR! NO FEATURES PASSED")
            return

        # создаем пустой датафрейм для ответа
        result = pd.DataFrame()

        # добавляем в него схожести по поисковикам
        ind = 1
        for searcher in self.searchers:
            sim = searcher.search(query, len(self.searchers[0].doc_ids), 0)
            if result.empty:
                result = pd.DataFrame(sim, columns=['doc_id', 'rel' + str(ind)])
            else:
                newDF = pd.DataFrame(sim, columns=['doc_id', 'rel' + str(ind)]).set_index('doc_id')
                result = result.join(newDF, on='doc_id')
            ind += 1

        self.df_to_pickle = result

        # добавляем финальную релевантность по заданной функции
        result['FinalRel'] = result.apply(self.relev_func, axis=1)
        result = result.sort_values(by='FinalRel', ascending=False)

        # возвращаем ответ в виде таблицы (отсортированной)
        if (dataFrReturned):
                return result.head(topN)

        return result['doc_id'].tolist()[:topN]

    # загрузка таблицы с признаками в path
    def save_df_features_to_pickle(self, path):
        if self.df_to_pickle is not None:
            self.df_to_pickle.to_pickle(path)

    # чтение таблицы с признаками
    @staticmethod
    def load_features(path):
        if os.path.exists(path):
            return pd.read_pickle(path)
