# Антон

from pymorphy2 import MorphAnalyzer
from collections import Counter
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from tools.inverse_index import InvIndex
from tools.tfidf import TFIDF
import tools.pravoved_recognizer as pravoved_recognizer


def print_ans(ans_arr, num_to_name, n):
    ans_arr.sort(key=lambda x: x[1], reverse=True)
    i = 1
    for el in ans_arr[:n]:
        print(i, "-ый по релевантности ответ:", end=' ')
        #print(num_to_name[el[0]])
        print(el[0])
        i += 1
    return ans_arr


class Search:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.cash = dict()
        self.morph = MorphAnalyzer()
        self.stop_words = stopwords.words("russian")


class Inv_Ind_Search(Search):
    # здесь inv_ind_file и т.д - путь до файлов (все файлы в формате pickle). Они уже существуют и их считать не надо
    def __init__(self, tokenizer, inverse_index_pickle):
        Search.__init__(self, tokenizer)
        self.inverse_index = InvIndex.load(inverse_index_pickle)
        self.cash = self.inverse_index.cash

    # обрабатывает запрос с клавиатуры и выдает к нему ответ
    # релевантность - ПО КОЛИЧЕСТВУ СЛОВ В ПЕРЕСЕЧЕНИИ (ОТВЕТА И ВОПРОСА)

    def request_processing_input_qRelev(self, inp):
        reqst = inp
        self.tokenizer.text = reqst
        reqst = self.tokenizer.tokenize(self.cash, self.morph, self.stop_words)
        ans = Counter()
        for word in reqst:
            if word in self.inverse_index.inv_ind:
                for el in self.inverse_index.inv_ind[word]:
                    ans[el[0]] += 1
        top = ans.most_common(5)
        i = 1
        for el in top:
            print(i, '-ый по релевантности ответ: ', self.inverse_index.num_to_name[el[0]], sep='')
            print("Текст статьи: ")
            print(self.inverse_index.num_to_text[el[0]])
            print()
            print()
            i += 1

    def request_processing_input_fmerRelev(self, inp):
        reqst = inp
        self.tokenizer.text = reqst
        reqst = self.tokenizer.tokenize(self.cash, self.morph, self.stop_words)
        help = Counter()
        ans = dict()
        for word in reqst:
            if word in self.inverse_index.inv_ind:
                for key in self.inverse_index.inv_ind[word]:
                    help[key[0]] += 1
        for key in list(self.inverse_index.num_to_text.keys()):
            Recall = help[key] / len(reqst)
            if not (self.inverse_index.num_tokens_dict[key]):
                F = -1000000000
            else:
                Presicion = help[key] / len(self.inverse_index.num_tokens_dict[key])
                if not help[key]:
                    F = -1000000000
                else:
                    F = 2 / ((1 / Presicion) + (1 / Recall))
            ans[key] = F
        list_ans = list(ans.items())
        print_ans(list_ans, self.inverse_index.num_to_name, 5)

    # функция предназначена для ответа на вопросы из правоведа
    # на каждый вопрос правоведа выдает топ-5 релевантных ответов
    # релевантность - ПО F - мере
    def request_processing_pravoved_fmerRelev(self, directory):
        pravoved = pravoved_recognizer.norms_codexes_to_normal(directory)
        for prav in pravoved:
            print(prav.question)
            self.request_processing_input_fmerRelev(prav.question)
            print("\n-------------------------------------\n")




class TFIDF_Search(Search):
    # класс с выделением признаков TF-IDF. Релевантность: косинусовая мера
    def __init__(self, tokenizer, tfidf_file):
        Search.__init__(self, tokenizer)
        self.tfidf_mod = TFIDF.load(tfidf_file)
        self.cash = self.tfidf_mod.inverse_index.cash

    def request_processing_input(self, inp):
        reqst = inp
        self.tokenizer.text = reqst
        reqst = self.tokenizer.tokenize(self.cash, self.morph, self.stop_words)
        req_tfidf_dict = self.tfidf_mod.request_counting([" ".join(reqst)])
        cos_sim = cosine_similarity(self.tfidf_mod.tfidf_matrix, req_tfidf_dict)

        ans = []
        i = 0
        for el in cos_sim:
            ans.append((self.tfidf_mod.num_to_num_dict[i], el[0]))
            i += 1

        ans = print_ans(ans, self.tfidf_mod.inverse_index.num_to_name, 5)
        return ans

    def request_tfidf(self, reqst):
        self.tokenizer.text = reqst
        reqst = self.tokenizer.tokenize(self.cash, self.morph, self.stop_words)
        req_tfidf_dict = self.tfidf_mod.request_counting([" ".join(reqst)])
        tfidf_vect = []
        for ind, data in zip(req_tfidf_dict.indices, req_tfidf_dict.data):
            tfidf_vect.append((ind, data))
        return sorted(tfidf_vect)

    def request_processing_input_without_print2(self, inp):
        reqst = inp
        self.tokenizer.text = reqst
        reqst = self.tokenizer.tokenize(self.cash, self.morph, self.stop_words)
        req_tfidf_dict = self.tfidf_mod.request_counting([" ".join(reqst)])
        cos_sim = cosine_similarity(self.tfidf_mod.tfidf_matrix, req_tfidf_dict)

        ans = []
        i = 0
        for el in cos_sim:
            ans.append((self.tfidf_mod.num_to_num_dict[i], el[0]))
            i += 1

        ans.sort(key=lambda x: x[1], reverse=True)
        return ans

    def cnt_cosine_similarity(self, reqst: pravoved_recognizer.Request):
        self.tokenizer.text = reqst
        reqst = self.tokenizer.tokenize(self.cash, self.morph, self.stop_words)
        req_tfidf_dict = self.tfidf_mod.request_counting([" ".join(reqst)])
        cos_sim = cosine_similarity(self.tfidf_mod.tfidf_matrix, req_tfidf_dict)
        return cos_sim

    def request_processing_input_without_print(self, inp, num_in_top):
        reqst = inp
        self.tokenizer.text = reqst
        reqst = self.tokenizer.tokenize(self.cash, self.morph, self.stop_words)
        req_tfidf_dict = self.tfidf_mod.request_counting([" ".join(reqst)])
        cos_sim = cosine_similarity(self.tfidf_mod.tfidf_matrix, req_tfidf_dict)

        ans = []
        i = 0
        for el in cos_sim:
            ans.append((self.tfidf_mod.num_to_num_dict[i], el[0]))
            i += 1

        ans.sort(key=lambda x: x[1], reverse=True)
        return ans[:num_in_top]


    # метод для обработки запроса из Правоведа
    def request_processing_pravoved(self, directory, num_in_top):
        pravoved = pravoved_recognizer.norms_codexes_to_normal(directory)
        for prav in pravoved:
            print(prav.question)
            self.request_processing_input(prav.question)
            print("\n-------------------------------------\n")
