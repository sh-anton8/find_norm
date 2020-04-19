# Антон

import pandas as pd

# базовый поисковик (агрегатор)
# searchers - массив поисковиков (например массив классов TFIDF)
class Baseline_Search():
    def __init__(self, tokenizer, searchers):
        self.tokenizer = tokenizer
        self.searchers = searchers

    @staticmethod
    def df_visualization(search_result):
        d = {}
        feature_name = 'Relev '
        fin_feature = 'Final relev'
        ids = [ans[0] for ans in search_result]
        for i in range(1, len(search_result[0])):
            if i != len(search_result[0]) - 1:
                d[feature_name + str(i)] = [ans[i] for ans in search_result]
            else:
                d[fin_feature] = [ans[i] for ans in search_result]

        df = pd.DataFrame(data=d, index=ids)
        return df


    # функция поиска (запрос, функция релевантности)
    # функция релевантонсти принимает массив признаков для i документа и возвращает его релевантность
    def search(self, query, relev_func, topN = 10):
        tokens = self.tokenizer.tokenize(query)

        # проверка, что передан зотя бы 1 поисковик
        if (len(self.searchers) == 0):
            print("ERROR! NO FEATURES PASSED")
            return

        # массив ответов
        # после работы i элемент содержит по порядку doc_id, схожесть i-го поисковика, финальную релевантность
        ans = [[doc_id] for doc_id in self.searchers[0].doc_ids]

        # добавляем схожести по поисковикам
        for i in range(len(self.searchers)):
            sim = self.searchers[i].similarity(tokens)
            for i in range(len(sim)):
                ans[i].append(sim[i])

        # добавляем финальную релевантность
        for i in range(len(ans)):
            ans[i].append(relev_func(ans[i][1:]))

        # ранжируем (по последнему аргументу (значению релевантности) и оставляем топ
        ans = sorted(ans, key=lambda arg: arg[-1], reverse=True)[:topN]

        # возвращаем ответ в виде таблицы (отсортированной)
        return self.df_visualization(ans)
