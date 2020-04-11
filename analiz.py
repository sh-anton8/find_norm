'''

INFO: Данный файл содержит класс Analizer для оценки качества модели

Описание: Данный класс позволяет посчитать и получить графики для следующих метрик:

- map@K
- nDCG@K
- mrr@K

А также получить график покрытия в зависимости от количества статей в топе

'''


import tools.search as search
import tools.pravoved_recognizer as pravoved_recognizer
import tools.tokenize_docs as tokenize_docs
import matplotlib.pyplot as plt
import math
import os

from tools.relative_paths_to_directories import path_to_directories

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25,\
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())

class Analizer:
    # sample - выборка (массив структур Запрос (request) из pravoved_recognizer.py)
    # answers - ответы модели (массив i эл-т которого - массив содержащий отсортированные по релевантности ответы на
    # i-ый запрос выборки в виде (номер в нашей нумерации, релевантность данной статьи))
    def __init__(self, answers, sample):
        self.sample = sample
        self.answers = answers

    @staticmethod
    def save_graphics(x, metric, ylabel: str, name_file: str):
        plt.plot(x, metric, color='red', label='Статьи')
        plt.ylabel(ylabel)
        plt.xlabel('Количество статей в топе')
        plt.legend()
        plt.savefig(os.path.join(PATH_TO_FILES, 'metrics_count', name_file))
        plt.show()

    # n - верхняя граница на количество статей в топе
    def top_n_cover(self, n, in_percent = True):

        x = []
        y_codex = []
        y_articles = []

        print(len(self.sample))


        for i in range(1, n, 2):
            codex = 0
            article = 0
            both = 0
            ind = 0
            for samp in self.sample:
                cod = 0
                art = 0
                answer = self.answers[ind][:i]
                ind += 1
                for ans in answer:
                    if ans[0][0] == samp.codex:
                        cod = 1
                    if ans[0][1] == samp.norm:
                        art = 1
                if cod == 1 and art == 1:
                    both += 1
                elif cod == 1:
                    codex += 1
                elif article == 1:
                    art += 1

            x.append(i)
            y_codex.append(codex + both)
            y_articles.append(both)

        if (not in_percent):
            plt.plot(x, y_codex, color='blue', label='Кодексы')
            plt.plot(x, y_articles, color='red', label='Статьи')
            plt.title('Покртие в зависимости от количества статей в топе')
            plt.ylabel('Попаданий')
            plt.xlabel('Количество статей в топе')
            plt.legend()
            plt.show()

        else:
            for i in range(len(y_articles)):
                y_articles[i] /= len(self.sample)
                y_codex[i] /= len(self.sample)
            plt.plot(x, y_codex, color='blue', label='Кодексы')
            plt.plot(x, y_articles, color='red', label='Статьи')
            plt.title('Покртие в зависимости от количества статей в топе')
            plt.ylabel('Попаданий (в процентах)')
            plt.xlabel('Количество статей в топе')
            plt.legend()
            plt.show()



    @staticmethod
    def ap_k(relev_positions, k):
        print(relev_positions, k)
        ans = 0
        num_rel = 0
        for rl in relev_positions:
            if rl < k:
                num_rel += 1
                ans += num_rel / rl
            else:
                break
        print(ans)
        return ans

    def map_k(self, K):
        apk = [0] * (K // 2)
        for j in range(len(self.sample)):
            actual_art = [(self.sample[j].codex, self.sample[j].norm)]
            predicted_art = []
            for ans in self.answers[j][:K]:
                predicted_art.append((ans[0][0], ans[0][1]))
            relev_positions = []
            for i, pa in enumerate(predicted_art):
                if pa in actual_art:
                    relev_positions.append(i + 1)
            for k in range(1, K, 2):
                apk[k // 2] += self.ap_k(relev_positions, k)
        apk = [a / len(self.sample) for a in apk]
        x = [i for i in range(1, K, 2)]
        print(apk)

        self.save_graphics(x=x, metric=apk, ylabel='MAP(k)', name_file='map.png')

    def ndcg(self, K):

        x = []
        y_articles = []

        for i in range(2, K, 2):
            sum_ndcg = 0
            for j in range(len(self.sample)):
                print(len(self.sample))
                answ = self.answers[j][:i]
                scores = []
                ndcg = 0
                for ans in answ:
                    print(ans[0][0], ans[0][1], self.sample[j].codex, self.sample[j].norm)
                    if (self.sample[j].codex, self.sample[j].norm) == (ans[0][0], ans[0][1]):
                        scores.append(1)
                    else:
                        scores.append(0)
                for k in range(1, len(scores) + 1):
                    ndcg += scores[k - 1] / math.log(k + 1, 2)
                sum_ndcg += ndcg
            print(sum_ndcg / len(self.sample))
            x.append(i)
            y_articles.append(sum_ndcg / len(self.sample))

        self.save_graphics(x=x, metric=y_articles, ylabel='NDCG', name_file='ndcg.png')




    @staticmethod
    def mrr_k(relev_positions, k):
        for rl in relev_positions:
            if rl < k:
                return 1 / rl
        return 0

    def mrr(self, K):
        mrr = [0] * (K // 2)
        for j in range(len(self.sample)):
            print(self.sample[j].answer)
            actual_art = [(self.sample[j].codex, self.sample[j].norm)]
            print(actual_art)
            predicted_art = []
            for ans in self.answers[j][:K]:
                predicted_art.append((ans[0][0], ans[0][1]))
            relev_positions = []
            for i, pa in enumerate(predicted_art):
                if pa in actual_art:
                    relev_positions.append(i + 1)
            for k in range(1, K, 2):
                mrr[k // 2] += self.mrr_k(relev_positions, k)
        mrr = [a / len(self.sample) for a in mrr]
        x = [i for i in range(1, K, 2)]
        print(mrr)

        self.save_graphics(x=x, metric=mrr, ylabel='MRR', name_file='mrr.png')


# Ниже приведен ПРИМЕР использования

'''

n = 5
tfidf_searcher = search.TFIDF_Search(tokenize_docs.Tokenizer('text'), "tf_idf/tf_idf_1")
pravoved = pravoved_recognizer.norms_codexes_to_normal("codexes")
ans_arr = [0] * len(pravoved)
for i in range(len(pravoved)):
    ans_arr[i] = tfidf_searcher.request_processing_input_without_print2(pravoved[i].question)

test = Analizer(ans_arr, pravoved)
test.top_n_cover(11, True)
test.map_k(11)
test.ndcg(11)
test.mrr(11)

'''
