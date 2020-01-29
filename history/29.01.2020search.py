# Антон

import tools.tokenize_docs as tokenize_docs
import pickle
from pymorphy2 import MorphAnalyzer
from collections import Counter
import tools.tfidf as tf
from nltk.corpus import stopwords
import tools.request_recognizer as request_recognizer



class Search:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.cash = dict()
        self.morph = MorphAnalyzer()
        self.stop_words = stopwords.words("russian")


class Inv_Ind_Search(Search):
    # здесь inv_ind_file и т.д - путь до файлов (все файлы в формате pickle). Они уже существуют и их считать не надо
    def __init__(self, tokenizer, inv_ind_file, ttn_dict_file, ntt_dict_file, n_to_tokens_file):
        Search.__init__(self, tokenizer)
        with open(ttn_dict_file, 'rb') as f:
            self.text_to_num = pickle.load(f)
        with open(ntt_dict_file, 'rb') as f:
            self.num_to_text = pickle.load(f)
        with open(inv_ind_file, 'rb') as f:
            self.inv_ind = pickle.load(f)
        with open(n_to_tokens_file, 'rb') as f:
            self.num_to_tokens = pickle.load(f)

    # обрабатывает запрос с клавиатуры и выдает к нему ответ
    # релевантность - ПО КОЛИЧЕСТВУ СЛОВ В ПЕРЕСЕЧЕНИИ (ОТВЕТА И ВОПРОСА)
    def request_processing_input_qRelev(self):
        reqst = input('Введите запрос: ')
        self.tokenizer.text = reqst
        reqst = self.tokenizer.tokenize(self.cash, self.morph, self.stop_words)
        ans = Counter()
        for word in reqst:
            if word in self.inv_ind:
                for el in self.inv_ind[word]:
                    ans[el[0]] += 1
        top = ans.most_common(5)
        i = 1
        for el in top:
            print(i, '-ый по релевантности ответ: ', el[0], sep='')
            print("Текст статьи: ")
            print(self.num_to_text[el[0]])
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
                if word in self.inv_ind:
                    for el in self.inv_ind[word]:
                        ans[el[0]] += 1
            top = ans.most_common(5)
            i = 1
            for el in top:
                if (i < 2):
                    print("-----------------------------------")
                    print(reqst.question)
                print(i, '-ый по релевантности ответ: ', el[0], sep='')
                i += 1
                if (i >= 6):
                    print("-----------------------------------")
                    print()

    # обрабатывает запрос с клавиатуры и выдает к нему ответ
    # релевантность - ПО F - мере
    def request_processing_input_fmerRelev(self):
        reqst = input('Введите запрос: ')
        self.tokenizer.text = reqst
        reqst = self.tokenizer.tokenize(self.cash, self.morph, self.stop_words)
        help = Counter()
        ans = dict()
        for word in reqst:
            if word in self.inv_ind:
                for key in self.inv_ind[word]:
                    help[key[0]] += 1
        for key in list(self.num_to_text.keys()):
            Recall = help[key] / len(reqst)
            if (len(self.num_to_tokens[key]) == 0):
                F = -100000000000000000000000000  # нужна помощь
            else:
                Presicion = help[key] / len(self.num_to_tokens[key])
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
            print(i, '-ый по релевантности ответ: ', el[0], sep='')
            print("Текст статьи: ")
            print(self.num_to_text[el[0]])
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
                if word in self.inv_ind:
                    for el in self.inv_ind[word]:
                        help[el[0]] += 1
            ans = dict()
            for key in list(self.num_to_text.keys()):
                Recall = help[key] / len(rqst)
                if (len(self.num_to_tokens[key]) == 0):
                    F = -100000000000000000000000000  # нужна помощь
                else:
                    Presicion = help[key] / len(self.num_to_tokens[key])
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
                print(i, '-ый по релевантности ответ: ', el[0], sep='')
                i += 1
                if (i >= 6):
                    print("-----------------------------------")
                    print()








class TFIDF_Search(Search):
    # класс с выделением признаков TF-IDF. Релевантность: косинусовая мера
    def __init__(self, tokenizer, tfidf_file):
        Search.__init__(self, tokenizer)
        with open(tfidf_file, 'rb') as f:
            self.tfidf_mod = pickle.load(f)

    # метод для обработки запроса с клавиатуры
    def request_processing_input(self):

        def scalar_mull(tfidf_dict1, tfidf_dict2):
            ans = 0
            for el in tfidf_dict1.keys():
                if el in tfidf_dict2:
                    ans += tfidf_dict1[el] * tfidf_dict2[el]
            return ans

        reqst = input('Введите запрос: ')
        self.tokenizer.text = reqst
        reqst = self.tokenizer.tokenize(self.cash, self.morph, self.stop_words)
        ans = []
        req_tfidf_dict = self.tfidf_mod.request_counting(reqst)

        for key in list(self.tfidf_mod.num_to_text.keys()):
            relev = scalar_mull(self.tfidf_mod.tfidf[key], req_tfidf_dict)
            a = scalar_mull(self.tfidf_mod.tfidf[key], self.tfidf_mod.tfidf[key])
            b = scalar_mull(req_tfidf_dict, req_tfidf_dict)
            a = a ** (1/2)
            b = b ** (1/2)

            if a == 0 or b == 0:
                ans.append((key, 0))
            else:
                ans.append((key, relev / (a * b)))

        ans.sort(key=lambda x: -x[1])
        top = ans[:5]
        i = 1
        for el in top:
            print(i, '-ый по релевантности ответ: ', el[0], sep='')
            print("Текст статьи: ")
            print(self.tfidf_mod.num_to_text[el[0]])
            print()
            print()
            i += 1

    # метод для обработки запроса из Правоведа
    def request_processing_pravoved(self, directory):

        def scalar_mull(tfidf_dict1, tfidf_dict2):
            ans = 0
            for el in tfidf_dict1.keys():
                if el in tfidf_dict2:
                    ans += tfidf_dict1[el] * tfidf_dict2[el]
            return ans


        s = request_recognizer.Separator(directory)
        requests = s.sep_by_requests()
        for requst in requests:
            self.tokenizer.text = requst.question
            reqst = self.tokenizer.tokenize(self.cash, self.morph, self.stop_words)
            ans = []
            req_tfidf_dict = self.tfidf_mod.request_counting(reqst)

            for key in list(self.tfidf_mod.num_to_text.keys()):
                relev = scalar_mull(self.tfidf_mod.tfidf[key], req_tfidf_dict)
                a = scalar_mull(self.tfidf_mod.tfidf[key], self.tfidf_mod.tfidf[key])
                b = scalar_mull(req_tfidf_dict, req_tfidf_dict)
                a = a ** (1 / 2)
                b = b ** (1 / 2)

                if a == 0 or b == 0:
                    ans.append((key, 0))
                else:
                    ans.append((key, relev / (a * b)))

            ans.sort(key=lambda x: -x[1])
            top = ans[:5]
            i = 1
            for el in top:
                if (i < 2):
                    print("-----------------------------------")
                    print(requst.question)
                print(i, '-ый по релевантности ответ: ', el[0], sep='')
                i += 1
                if (i >= 6):
                    print("-----------------------------------")
                    print()




# ПОИСК ПО ОБРАТНОМУ ИНДЕКСУ

# Пример того, как можно посчитать и сохранить в файл Обратный индекс и сопутствующие файлы
# (а именно словарь - номер статьи -> ее текст и в обратную сторону, а также словарь номер статьи - токены)


'''
dir = "CodexAndLowTexts"
s = ii.InvIndex(dir, tokenize_docs.Tokenizer('text'))
s.update_dicts('chapter')
s.build_inversed_index('chapter')
s.num_tokens_dict_builder()
s.save_tokens_dict("num_to_tokens_dict_codexes_for_chapter")
s.save_index("inv_ind_codexes_for_chapter")
s.save_ntt_dict("num_to_text_dict_codexes_for_chapter")
s.save_ttn_dict("text_to_num_dict_codexes_for_chapter")
s.save_ntl_dict("num_to_len_dict_codexes_for_chapter")
'''

# Все сохраненные файлы сохранены на Я.ДИСК
# Ссылка на папку с файлами: https://yadi.sk/d/iifkA77f5Tv8Bg
# Названия на диске совпадают с названиями в скобках


# Пример поиска основанного на обратном индексе
# request_processing_input_qRelev() - отвечает на запрос с клавиатуры
# request_processing_pravoved_fmerRelev(dir2) - отвечает за ответ на вопросы с правоведа

'''

searcher = Inv_Ind_Search(tokenize_docs.Tokenizer('text'),
                         "files/inv_ind_codexes_for_chapter",
                          "files/text_to_num_dict_codexes_for_chapter",
                          "files/num_to_text_dict_codexes_for_chapter",
                          "files/num_to_tokens_dict_codexes_for_chapter")
# searcher.request_processing_input_qRelev()
dir2 = "files/pravoved_articles.txt"
searcher.request_processing_pravoved_fmerRelev(dir2)


'''










# ПОИСК ПО TF-IDF


# пример предварительного создания класса tfidf с целью дальнейшего поиска с его использованием

'''
mod = tf.tfidf("files/text_to_num_dict_codexes_for_chapter",
            "files/num_to_text_dict_codexes_for_chapter",
            "files/num_to_tokens_dict_codexes_for_chapter",
            "files/inv_ind_codexes_for_chapter",
            "files/num_to_len_dict_codexes_for_chapter")

mod.count_tf_idf()
with open("tfidf_codexes_for_chapter", 'wb') as f:
    pickle.dump(mod, f)

'''

# Все файлы сохранены на Я.ДИСК
# Ссылка на папку с файлами: https://yadi.sk/d/iifkA77f5Tv8Bg
# Названия на диске совпадают с названиями в скобках



# Ниже приведен пример работы поиска как с клавиатуры, так и для запросов из правоведа
'''

tfidf_searcher = TFIDF_Search(tokenize_docs.Tokenizer('text'), "files/tfidf_codexes_for_chapter")
# tfidf_searcher.request_processing_input()
dir2 = "files/pravoved_articles.txt"
tfidf_searcher.request_processing_pravoved(dir2)

'''
