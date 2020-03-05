#Настя

import re
import os


chp_sep_word = "Глава [\d\.'I''V''X']* "
art_sep_word = "Статья [\d\.]* "
par_sep_word = "\n\d\. "
art_sep_word_name = 'Статья .*?\n'

class Collection:  #класс для коллекции
    def __init__(self, text):
        self.text = text
        self.d = {}


    def itersplit(self, sep):  #функция для разделения текста по sep и возвращаю по одному
        exp = re.compile(sep)
        if exp.search(self.text, 0) is None:
            return
        pos = 0
        m1 = exp.search(self.text, 0)
        while True:
            m = exp.search(self.text, pos)
            if not m:
                yield self.text[pos:], m1[0]
                break
            yield self.text[pos:m.start()], m1[0]
            pos = m.end()
            m1 = m


def iter_by_chapter(collect, doc_id): #итерирование по главам с возвращением двух словрей -- d[текст] = номер главы,
                                        # d[номер главы] = текст
    d_chp, d_rev = {}, {}
    for i, num_chp in collect.itersplit(chp_sep_word):
        num_chp = num_chp.split(' ')[1]
        d_chp[doc_id, num_chp] = i
        d_rev[i] = (doc_id, num_chp)
    return d_chp, d_rev


def iter_by_art_no_chp(collect, doc_id):
    d_art, d_rev = {}, {}
    for i, num_chp in collect.itersplit(chp_sep_word):
        new_col = Collection(i)
        for j, num_art in new_col.itersplit(art_sep_word):
            num_art = num_art.split(' ')[1]
            d_art[(doc_id, num_art)] = j
            d_rev[j] = (doc_id, num_art)
    return d_art, d_rev


def iter_by_art(collect, doc_id):  #возвращение словарей с номером главы и статьи
    d_art, d_rev = {}, {}
    for i, num_chp in collect.itersplit(chp_sep_word):
        num_chp = num_chp.split(' ')[1]
        new_col = Collection(i)
        for j, num_art in new_col.itersplit(art_sep_word):
            num_art = num_art.split(' ')[1]
            d_art[(doc_id, num_chp[:-1], num_art[:-1])] = j
            d_rev[j] = (doc_id, num_chp[:-1], num_art[:-1])
    return d_art, d_rev


def art_name(collect, doc_id):
    name = {}
    for i, num_chp in collect.itersplit(chp_sep_word):
        new_col = Collection(i)
        num_chp = num_chp.split(' ')[1]
        for j, num_art in new_col.itersplit(art_sep_word_name):
            name[(doc_id, num_chp[:-1], num_art.split(' ')[1][:-1])] = ' '.join(num_art.split()[2:])
    return name

def iter_by_par(collect, doc_id):  #возвращение словарей с номером главы, статьи и пункта
    d_par, d_rev = {}, {}
    for i, num_chp in collect.itersplit(chp_sep_word):
        new_col = Collection(i)
        num_chp = num_chp.split(' ')[1]
        for j, num_art in new_col.itersplit(art_sep_word):
            num_art = num_art.split(' ')[1]
            new_col2 = Collection(j)
            for f, num_par in new_col2.itersplit(par_sep_word):
                num_par = re.search('[\d\.]', num_par)[0]
                d_par[(doc_id, num_chp, num_art, num_par)] = f
                d_rev[f] = (doc_id, num_chp, num_art, num_par)
    return d_par, d_rev


def chp(col): #итерирование по главам
    for i, num_chp in col.itersplit(chp_sep_word):
        yield i

def art(col): #итерирование по статьям
    for i, num_chp in col.itersplit(chp_sep_word):
        new_col = Collection(i)
        for j, num_art in new_col.itersplit(art_sep_word):
            yield j


def par(col): #итерирование по пунктам
    for i, num_chp in col.itersplit(chp_sep_word):
        new_col = Collection(i)
        for j, num_art in new_col.itersplit(art_sep_word):
            new_col2 = Collection(j)
            for f, num_par in new_col2.itersplit(par_sep_word):
                yield f


def iter_by_docs(docs, dir, iter_by, it): #итерирование по документам, в параметрах по чему итерироваться(iter_by), и что именно нужно сделать при итерировании -- вернуть словарь(it=0) или постепенно каждую часть(it=1)
    doc_id = docs[docs.find('_') + 1: docs.find('.')]
    f = open(dir + '/' + docs, 'r', encoding='utf-8')
    file = f.read()
    c = Collection(file)
    if it == 0:
        if iter_by == 'chapter':
            return iter_by_chapter(c, doc_id)
        elif iter_by == 'paragraph':
            return iter_by_par(c, doc_id)
        elif iter_by == 'article':
            return iter_by_art(c, doc_id)
        elif iter_by == 'art_no_chp':
            return iter_by_art_no_chp(c, doc_id)
        else:
            print("MISTAKE")
            return
    else:
        if iter_by == 'chapter':
            return chp(c)
        elif iter_by == 'paragraph':
            return par(c)
        elif iter_by == 'article':
            return art(c)
        elif iter_by == 'art_name':
            return art_name(c, doc_id)
        else:
            print("MISTAKE")
            return


def iter_pravoved(docs): #итерирование по документам, в параметрах по чему итерироваться(iter_by), и что именно нужно сделать при итерировании -- вернуть словарь(it=0) или постепенно каждую часть(it=1)
    doc_id = docs[docs.find('_') + 1: docs.find('.')]
    f = open(docs, 'r', encoding='utf-8')
    file = f.read()
    c = Collection(file)
    return iter_by_art_no_chp(c, doc_id)


'''
names = iter_by_docs("codex_1.txt", "codexes", 'art_name', 1)       #Возвращает словарь, где ключ -- пара (документ, статья), значение  -- название статьи
print(names.keys())

'''



'''
file = open("codexes/codex_1.txt", 'r')
a, b = iter_by_docs("codex_2.txt", "codexes", 'article', 0)
print(a.keys())

f = file.read()
coll = Collection(f)

J, Z = iter_by_chapter(coll, 1)
print(J.keys())
for k in par(coll):
    print(k)
    print('---')

'''

