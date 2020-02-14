# Антон

import tools.tokenize_docs as tokenize_docs
import pickle
from pymorphy2 import MorphAnalyzer
from collections import Counter
import tools.tfidf as tf
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import tools.requests_recognizer as request_recognizer
import tools.inverse_index as ii
import numpy as np
import tools.tfidf as tfidf
import tools.coll as coll
import tools.name_codexes as names_cod
import tools.pravoved_recognizer as pravoved_recognizer
from scipy.sparse import csr_matrix


def print_ans(ans_arr, num_to_name, num_to_texr, n):
    ans_arr.sort(key=lambda x: x[1], reverse=True)
    i = 1
    for el in ans_arr[:n]:
        print(i, "-ый по релевантности ответ:", end=' ')
        print(num_to_name[el[0]])
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
        self.inverse_index = ii.load_inverse_index(inverse_index_pickle)
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
            if (len(self.inverse_index.num_tokens_dict[key]) == 0):
                F = -1000000000
            else:
                Presicion = help[key] / len(self.inverse_index.num_tokens_dict[key])
                if (help[key] == 0):
                    F = -1000000000
                else:
                    F = 2 / ((1 / Presicion) + (1 / Recall))
            ans[key] = F
        list_ans = list(ans.items())
        print_ans(list_ans, self.inverse_index.num_to_name, self.inverse_index.num_to_text, 5)

    # функция предназначена для ответа на вопросы из правоведа
    # на каждый вопрос правоведа выдает топ-5 релевантных ответов
    # релевантность - ПО F - мере
    def request_processing_pravoved_fmerRelev(self, directory):
        pravoved = pravoved_recognizer.norms_codexes_to_normal(directory)
        for prav in pravoved:
            print(prav.question)
            self.request_processing_input_fmerRelev(prav.question)
            print()
            print("-------------------------------------")
            print()








class TFIDF_Search(Search):
    # класс с выделением признаков TF-IDF. Релевантность: косинусовая мера
    def __init__(self, tokenizer, tfidf_file):
        Search.__init__(self, tokenizer)
        self.tfidf_mod = tfidf.load_tfidf(tfidf_file)
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

        ans = print_ans(ans, self.tfidf_mod.inverse_index.num_to_name, self.tfidf_mod.inverse_index.num_to_text, 5)
        return ans


    # метод для обработки запроса из Правоведа
    def request_processing_pravoved(self, directory):
        pravoved = pravoved_recognizer.norms_codexes_to_normal(directory)
        for prav in pravoved:
            print(prav.question)
            self.request_processing_input(prav.question)
            print()
            print("-------------------------------------")
            print()




# ПОИСК ПО ОБРАТНОМУ ИНДЕКСУ

# Пример того, как можно посчитать и сохранить в файл Обратный индекс и сопутствующие файлы
# (а именно словарь - номер статьи -> ее текст и в обратную сторону, а также словарь номер статьи - токены)

'''
dir = "codexes"
s = ii.InvIndex(dir, tokenize_docs.Tokenizer('text'))
s.update_dicts('article')
s.build_inversed_index('article')
s.num_tokens_dict_builder()
ii.save_inverse_index(s, "inv_ind_codexes_for_article_pickle")
'''

# Все сохраненные файлы сохранены на Я.ДИСК
# Ссылка на папку с файлами: https://yadi.sk/d/iifkA77f5Tv8Bg
# Названия на диске совпадают с названиями в скобках


# Пример поиска основанного на обратном индексе
# request_processing_input_qRelev() - отвечает на запрос с клавиатуры
# request_processing_pravoved_fmerRelev(dir2) - отвечает за ответ на вопросы с правоведа

'''
searcher = Inv_Ind_Search(tokenize_docs.Tokenizer('text'),
                         "files/inv_ind_codexes_for_article_pickle")
# searcher.request_processing_input_qRelev()
dir2 = "files/pravoved_articles.txt"
searcher.request_processing_pravoved_fmerRelev("codexes")
'''










# ПОИСК ПО TF-IDF


# пример предварительного создания класса tfidf с целью дальнейшего поиска с его использованием

'''
mod = tf.tfidf("files/inv_ind_codexes_for_article_pickle", (1, 1))
mod.count_tf_idf()
tfidf.save_tfidf(mod, "tfidf(1_1)_codexes_for_article_pickle")
'''


# Все файлы сохранены на Я.ДИСК
# Ссылка на папку с файлами: https://yadi.sk/d/iifkA77f5Tv8Bg
# Названия на диске совпадают с названиями в скобках



# Ниже приведен пример работы поиска как с клавиатуры, так и для запросов из правоведа

'''
tfidf_searcher = TFIDF_Search(tokenize_docs.Tokenizer('text'), "files/tfidf(1_1)_codexes_for_article_pickle")
pravoved = pravoved_recognizer.norms_codexes_to_normal("codexes")
codex = 0
article = 0
both = 0 #из 100 запросов совпали и кодекс, и статья в 16 случаях, только кодекс в 48
for pr in pravoved:
    cod = 0
    art = 0
    answers = tfidf_searcher.request_processing_input(pr.question)
    for ans in answers:
        if ans[0][0] == pr.codex:
            cod = 1
        if ans[0][2] == pr.norm:
            art = 1
    if cod == 1 and art == 1:
        both += 1
    elif cod == 1:
        codex += 1
    elif article == 1:
        art += 1

print("codex: ", codex)
print("article: ", article)
print("both", both)
'''


'''
tfidf_searcher = TFIDF_Search(tokenize_docs.Tokenizer('text'), "files/tfidf(1_1)_codexes_for_article_pickle")
tfidf_searcher.request_processing_input('Со скольки лет можно ездить детям без детского сидения')
'''


'''
tfidf_searcher = TFIDF_Search(tokenize_docs.Tokenizer('text'), "files/tfidf(1_1)_codexes_for_article_pickle")
dir2 = "files/pravoved_articles.txt"
tfidf_searcher.request_processing_pravoved("codexes")
'''