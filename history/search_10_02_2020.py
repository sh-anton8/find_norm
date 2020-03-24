# Антон

import tools.tokenize_docs as tokenize_docs
import pickle
from pymorphy2 import MorphAnalyzer
from collections import Counter
import tools.tfidf as tf
from nltk.corpus import stopwords
import tools.inverse_index as ii
import numpy as np
import tools.tfidf as tfidf
import tools.coll as coll
import tools.name_codexes as names_cod
import tools.pravoved_recognizer as pravoved_recognizer



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
    def request_processing_keyboardinput_qRelev(self):
        reqst = input('Введите запрос: ')
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
            print(el[0])
            print("Текст статьи: ")
            print(self.inverse_index.num_to_text[el[0]])
            print()
            print()
            i += 1

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

    # функция предназначена для ответа на вопросы из правоведа
    # на каждый вопрос правоведа выдает топ-5 релевантных ответов
    # релевантность - ПО КОЛИЧЕСТВУ СЛОВ В ПЕРЕСЕЧЕНИИ (ОТВЕТА И ВОПРОСА)
    def request_processing_pravoved_qRelev(self, directory):
        s = request_recognizer.Separator(directory)
        requests = s.sep_by_requests()
        for reqst in requests:
            self.tokenizer.text = reqst.question
            rqst = self.tokenizer.tokenize(self.cash, self.morph, self.stop_words)
            ans = Counter()
            for word in rqst:
                if word in self.inverse_index.inv_ind:
                    for el in self.inverse_index.inv_ind[word]:
                        ans[el[0]] += 1
            top = ans.most_common(5)
            i = 1
            for el in top:
                if (i < 2):
                    print("-----------------------------------")
                    print(reqst.question)
                print(i, '-ый по релевантности ответ: ', self.inverse_index.num_to_name[el[0]], sep='')
                i += 1
                if (i >= 6):
                    print("-----------------------------------")
                    print()

    # обрабатывает запрос с клавиатуры и выдает к нему ответ
    # релевантность - ПО F - мере
    def request_processing_keyboardinput_fmerRelev(self):
        reqst = input('Введите запрос: ')
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
                F = -100000000000000000000000000  # нужна помощь
            else:
                Presicion = help[key] / len(self.inverse_index.num_to_tokens[key])
                if (help[key] == 0):
                    F = -100000000000000000000000000
                else:
                    F = 2 / ((1 / Presicion) + (1 / Recall))
            ans[key] = F
        list_ans = list(ans.items())
        list_ans.sort(key=lambda x: -x[1])
        top = list_ans[:5]
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
                F = -100000000000000000000000000  # нужна помощь
            else:
                Presicion = help[key] / len(self.inverse_index.num_to_tokens[key])
                if (help[key] == 0):
                    F = -100000000000000000000000000
                else:
                    F = 2 / ((1 / Presicion) + (1 / Recall))
            ans[key] = F
        list_ans = list(ans.items())
        list_ans.sort(key=lambda x: -x[1])
        top = list_ans[:5]
        i = 1
        for el in top:
            print(i, '-ый по релевантности ответ: ', self.inverse_index.num_to_name[el[0]], sep='')
            print("Текст статьи: ")
            print(self.inverse_index.num_to_text[el[0]])
            print()
            print()
            i += 1

    # функция предназначена для ответа на вопросы из правоведа
    # на каждый вопрос правоведа выдает топ-5 релевантных ответов
    # релевантность - ПО F - мере
    def request_processing_pravoved_fmerRelev(self, directory):
        s = request_recognizer.Separator(directory)
        requests = s.sep_by_requests()
        for reqst in requests:
            self.tokenizer.text = reqst.question
            rqst = self.tokenizer.tokenize(self.cash, self.morph, self.stop_words)
            help = Counter()
            for word in rqst:
                if word in self.inverse_index.inv_ind:
                    for el in self.inverse_index.inv_ind[word]:
                        help[el[0]] += 1
            ans = dict()
            for key in list(self.inverse_index.num_to_text.keys()):
                Recall = help[key] / len(rqst)
                if (len(self.inverse_index.num_tokens_dict[key]) == 0):
                    F = -100000000000000000000000000  # нужна помощь
                else:
                    Presicion = help[key] / len(self.inverse_index.num_tokens_dict[key])
                    if (help[key] == 0):
                        F = -100000000000000000000000000
                    else:
                        F = 2 / ((1 / Presicion) + (1 / Recall))
                ans[key] = F
            list_ans = list(ans.items())
            list_ans.sort(key=lambda x: -x[1])
            top = list_ans[:5]
            i = 1
            for el in top:
                if (i < 2):
                    print("-----------------------------------")
                    print(reqst.question)
                print(i, '-ый по релевантности ответ: ', self.inverse_index.num_to_name[el[0]], sep='')
                i += 1
                if (i >= 6):
                    print("-----------------------------------")
                    print()








class TFIDF_Search(Search):
    # класс с выделением признаков TF-IDF. Релевантность: косинусовая мера
    def __init__(self, tokenizer, tfidf_file):
        Search.__init__(self, tokenizer)
        self.tfidf_mod = tfidf.load_tfidf(tfidf_file)
        self.cash = self.tfidf_mod.inverse_index.cash

    # метод для обработки запроса с клавиатуры
    def request_processing_keyboard_input(self):
        reqst = input('Введите запрос: ')
        self.tokenizer.text = reqst
        reqst = self.tokenizer.tokenize(self.cash, self.morph, self.stop_words)
        ans = []
        req_tfidf_dict = self.tfidf_mod.request_counting([" ".join(reqst)])

        for i in list(self.tfidf_mod.num_to_num_dict.keys()):
            relev = np.dot(self.tfidf_mod.tfidf_matrix[i, :].todense(), req_tfidf_dict.todense().transpose())
            a = np.dot(self.tfidf_mod.tfidf_matrix[i, :].todense(),
                       self.tfidf_mod.tfidf_matrix[i, :].todense().transpose())
            b = np.dot(req_tfidf_dict.todense(), req_tfidf_dict.todense().transpose())
            a = np.sqrt(a)
            b = np.sqrt(b)

            if a == 0 or b == 0:
                ans.append((self.tfidf_mod.num_to_num_dict[i], 0))
            else:
                ans.append((self.tfidf_mod.num_to_num_dict[i], relev / (a * b)))

        ans.sort(key=lambda x: -x[1])
        top = ans[:5]
        i = 1
        for el in top:
            print(i, '-ый по релевантности ответ: ', self.tfidf_mod.inverse_index.num_to_name[el[0]], sep='')
            print("Текст статьи: ")
            print(self.tfidf_mod.inverse_index.num_to_text[el[0]])
            print()
            print()
            i += 1

    def request_processing_input(self, inp):
        ans_req = []
        reqst = inp
        self.tokenizer.text = reqst
        reqst = self.tokenizer.tokenize(self.cash, self.morph, self.stop_words)
        ans = []
        req_tfidf_dict = self.tfidf_mod.request_counting([" ".join(reqst)])

        for i in list(self.tfidf_mod.num_to_num_dict.keys()):
            relev = np.dot(self.tfidf_mod.tfidf_matrix[i, :].todense(), req_tfidf_dict.todense().transpose())
            a = np.dot(self.tfidf_mod.tfidf_matrix[i, :].todense(),
                       self.tfidf_mod.tfidf_matrix[i, :].todense().transpose())
            b = np.dot(req_tfidf_dict.todense(), req_tfidf_dict.todense().transpose())
            a = np.sqrt(a)
            b = np.sqrt(b)

            if a == 0 or b == 0:
                ans.append((self.tfidf_mod.num_to_num_dict[i], 0))
            else:
                ans.append((self.tfidf_mod.num_to_num_dict[i], relev / (a * b)))

        ans.sort(key=lambda x: -x[1])
        top = ans[:5]
        i = 1
        for el in top:
            print(i, '-ый по релевантности ответ: ', names_cod.name_codexes[int(el[0][0])], ', ',
                  self.tfidf_mod.inverse_index.num_to_name[el[0]], sep='')
            print("Текст статьи: ")
            print(self.tfidf_mod.inverse_index.num_to_text[el[0]])
            print()
            print()
            i += 1
            ans_req.append(el[0])
        return ans_req

    # метод для обработки запроса из Правоведа
    def request_processing_pravoved(self, directory):
        s = request_recognizer.Separator(directory)
        requests = s.sep_by_requests()
        for requst in requests:
            self.tokenizer.text = requst.question
            reqst = self.tokenizer.tokenize(self.cash, self.morph, self.stop_words)
            ans = []
            req_tfidf_dict = self.tfidf_mod.request_counting([" ".join(reqst)])

            for i in list(self.tfidf_mod.num_to_num_dict.keys()):
                relev = np.dot(self.tfidf_mod.tfidf_matrix[i, :].todense(), req_tfidf_dict.todense().transpose())
                a = np.dot(self.tfidf_mod.tfidf_matrix[i, :].todense(), self.tfidf_mod.tfidf_matrix[i, :].todense().transpose())
                b = np.dot(req_tfidf_dict.todense(), req_tfidf_dict.todense().transpose())
                a = np.sqrt(a)
                b = np.sqrt(b)

                if a == 0 or b == 0:
                    ans.append((self.tfidf_mod.num_to_num_dict[i], 0))
                else:
                    ans.append((self.tfidf_mod.num_to_num_dict[i], relev / (a * b)))

            ans.sort(key=lambda x: -x[1])
            top = ans[:5]
            i = 1
            for el in top:
                if (i < 2):
                    print("-----------------------------------")
                    print(requst.question)
                print(i, '-ый по релевантности ответ: ', self.tfidf_mod.inverse_index.num_to_name[el[0]], sep='')
                i += 1
                if (i >= 6):
                    print("-----------------------------------")
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
searcher.request_processing_pravoved_fmerRelev(dir2)


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
        if ans[0] == pr.codex:
            cod = 1
        if ans[2] == pr.norm:
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
print(pravoved[0])
tfidf_searcher.request_processing_input('развод')
'''
'''
dir2 = "files/pravoved_articles.txt"
tfidf_searcher.request_processing_pravoved(dir2)

'''
