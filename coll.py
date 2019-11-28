import re

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


def iter_by_chapter(collect): #итерирование по главам с возвращением двух словрей -- d[текст] = номер главы,
                                        # d[номер главы] = текст
    d_chp, d_rev = {}, {}
    for i, num_chp in collect.itersplit(chp_sep_word):
        num_chp = num_chp.split(' ')[1]
        d_chp[num_chp] = i
        d_rev[i] = num_chp
    return d_chp, d_rev


def iter_by_art(collect):  #возвращение словарей с номером главы и статьи
    d_art, d_rev = {}, {}
    for i, num_chp in collect.itersplit(chp_sep_word):
        num_chp = num_chp.split(' ')[1]
        new_col = Collection(i)
        for j, num_art in new_col.itersplit(art_sep_word):
            num_art = num_art.split(' ')[1]
            d_art[(num_chp, num_art)] = j
            d_rev[j] = (num_chp, num_art)

    return d_art, d_rev

def iter_by_par(collect):  #возвращение словарей с номером главы, статьи и пункта
    d_par, d_rev = {}, {}
    for i, num_chp in collect.itersplit(chp_sep_word):
        new_col = Collection(i)
        num_chp = num_chp.split(' ')[1]
        for j, num_art in new_col.itersplit(art_sep_word):
            num_art = num_art.split(' ')[1]
            new_col2 = Collection(j)
            for f, num_par in new_col2.itersplit(par_sep_word):
                num_par = re.search('[\d\.]', num_par)[0]
                d_par[(num_chp, num_art, num_par)] = f
                d_rev[f] = (num_chp, num_art, num_par)
    return d_par, d_rev


def iter_by_docs(docs): #проход по всем документом с последующим итерированием по пунктам
    for d in docs:
        c = Collection(d)
        iter_by_par(c)


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


file = open("codex_1.txt", 'r')
f = file.read()
coll = Collection(f)
chp_sep_word = "Глава [\d\.]* "
art_sep_word = "Статья [\d\.]* "
par_sep_word = "\n\d\. "

J, Z = iter_by_chapter(coll)
print(J.keys())

'''
for k in par(coll):
    print(k)
    print('---')
'''

