'''

INFO: Данный файл содержит класс ExpAnalizer для оценки качества модели

Описание: В отличии от обычного анализа, теперь можно задать эпсилон окресность правильных статей
Т.Е теперь правильная статья не 1, а 1 + 2 * epsilon

'''


import matplotlib.pyplot as plt
import math
import os
from tools.inverse_index import InvIndex
from tools.relative_paths_to_directories import path_to_directories

PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25,\
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())


class ExpAnalizer:
    # sample - выборка (массив структур Запрос (request) из pravoved_recognizer.py)
    # answers - ответы модели (массив i эл-т которого - массив содержащий отсортированные по релевантности ответы на
    # i-ый запрос выборки в виде (номер в нашей нумерации))
    # epsilon - параметр Epsilon (см. описание)
    def __init__(self, answers, sample, epsilon):
        self.sample = sample
        self.answers = answers
        self.epsilon = epsilon
        self.id_to_num = {}
        invInd = InvIndex.load(PATH_TO_INV_IND)
        self.num_to_id = {}
        for doc_num, doc_id in enumerate(invInd.doc_ids):
            self.id_to_num[doc_id] = doc_num
            self.num_to_id[doc_num] = doc_id

    @staticmethod
    def save_graphics(x, metric, ylabel: str, name_file: str):
        plt.plot(x, metric, color='red', label='Статьи')
        plt.ylabel(ylabel)
        plt.xlabel('Количество статей в топе')
        plt.legend()
        plt.savefig(os.path.join(PATH_TO_FILES, 'metrics_count', f'{name_file}_exper.png'))
        plt.show()

    # n - верхняя граница на количество статей в топе
    def top_n_cover(self, n, in_percent=True):

        x = []
        y_codex = []
        y_articles = []

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
                    if ans[0] == samp['codex']:
                        cod = 1
                    right_ans = (str(samp['codex']), samp['norm'])
                    if self.id_to_num.get(right_ans, -100) - self.epsilon <= self.id_to_num[(ans)]\
                            <= self.id_to_num.get(right_ans, -100) + self.epsilon:
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
        ans = 0
        num_rel = 0
        for rl in relev_positions:
            if rl <= k:
                num_rel += 1
                ans += num_rel / rl
                return ans
            else:
                break
        return ans

    def map_k(self, K):
        apk = [0] * ((K + 1) // 2)
        for j in range(len(self.sample)):
            right_ans = (str(self.sample[j]['codex']), self.sample[j]['norm'])
            right_ans_num = self.id_to_num.get(right_ans, -100)
            actual_art = [self.num_to_id.get(ind, (-1000, -1000)) for ind in
                          range(right_ans_num - self.epsilon, right_ans_num + self.epsilon + 1)]
            predicted_art = []
            for ans in self.answers[j][:K]:
                predicted_art.append((ans[0], ans[1]))
            relev_positions = []
            for i, pa in enumerate(predicted_art):
                if pa in actual_art:
                    relev_positions.append(i + 1)
            for k in range(1, K + 1, 2):
                apk[k // 2] += self.ap_k(relev_positions, k)
        apk = [a / len(self.sample) for a in apk]
        x = [i for i in range(1, K + 1, 2)]
        print("exper_map: ", apk)
        self.save_graphics(x=x, metric=apk, ylabel='MAP(k)', name_file='map')

    def ndcg(self, K):
        ndcg = [0] * ((K + 1) // 2)
        for j in range(len(self.sample)):
            right_ans = (str(self.sample[j]['codex']), self.sample[j]['norm'])
            right_ans_num = self.id_to_num.get(right_ans, -100)
            actual_art = [self.num_to_id.get(ind, (-1000, -1000)) for ind in
                          range(right_ans_num - self.epsilon, right_ans_num + self.epsilon + 1)]
            predicted_art = []
            for ans in self.answers[j][:K]:
                predicted_art.append((str(ans[0]), ans[1]))
            relev_positions = []
            for i, pa in enumerate(predicted_art):
                if pa in actual_art:
                    relev_positions.append(i + 1)
            for k in range(1, K + 1, 2):
                for r in relev_positions:
                    if r <= k:
                        ndcg[k // 2] += 1/math.log(r + 1, 2)
                        break
        ndcg = [a / len(self.sample) for a in ndcg]
        print("exper_ndcg: ",  ndcg)
        x = [i for i in range(1, K + 1, 2)]
        self.save_graphics(x=x, metric=ndcg, ylabel='NDCG(k)', name_file='ndcg')




    @staticmethod
    def mrr_k(relev_positions, k):
        for rl in relev_positions:
            if rl <= k:
                return 1 / rl
        return 0

    def mrr(self, K):
        mrr = [0] * ((K + 1) // 2)
        for j in range(len(self.sample)):
            right_ans = (str(self.sample[j]['codex']), self.sample[j]['norm'])
            right_ans_num = self.id_to_num.get(right_ans, -100)
            actual_art = [self.num_to_id.get(ind, (-1000, -1000)) for ind in
                          range(right_ans_num - self.epsilon, right_ans_num + self.epsilon + 1)]
            predicted_art = []
            for ans in self.answers[j][:K]:
                predicted_art.append((ans[0], ans[1]))
            relev_positions = []
            for i, pa in enumerate(predicted_art):
                if pa in actual_art:
                    relev_positions.append(i + 1)
            for k in range(1, K + 1, 2):
                mrr[k // 2] += self.mrr_k(relev_positions, k)
        mrr = [a / len(self.sample) for a in mrr]
        x = [i for i in range(1, K + 1, 2)]
        print("exper_mrr:", mrr)

        self.save_graphics(x=x, metric=mrr, ylabel='MRR', name_file='mrr')

