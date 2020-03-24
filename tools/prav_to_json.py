#Настя

import re
import tools.coll as coll
import os
import json
import tools.name_codexes as nc

class Request:
    def __init__(self, theme, question, answer):
        self.theme = theme
        self.question = question
        self.answer = answer
        self.norm = []
        self.codex = []
        self.cod_norm = []

    def __str__(self):
        return 'question:{}\n\nanswer:{}\n\nnorm: {}\ncodex: {}\n-------------\n'\
            .format(self.question, self.answer, self.norm, self.codex)



class Separator:
    def __init__(self, path_to_file):
        self.path_to_file = path_to_file

    def editor(self):                   #деление текста на запросы
        file = open(self.path_to_file, 'r', encoding='utf-8')
        file = file.read().lower()
        file = file.replace('\xa0', "")
        a = re.split('\n(?!\t)', file)
        return a

    def sep_by_requests(self):    #деление запросов на тема-вопрос-ответ
        ed = self.editor()      #возвращает массив классов Request
        ans = []
        for k in ed:
            x1 = k.find('\t')
            x2 = k.find('\t\t', x1)
            x3 = k.find('\t\t\t', x2)
            if x3 == -1:
                s = Request(k[:x1], k[x1 + 1: x2], k[x2 + 1:])
                ans.append(s)
        return ans

    def list_of_codexes(self):  #возвращает распространненые аббревиатуры кодексов
        codex_names = ['УИК ', 'КАС ', 'СК ', 'ЖК ', 'АПК', 'УПК ', 'ГПК ', 'ук ', 'НК ', 'ТК ', 'ГК ', 'КоАП', 'зк '
                       , 'бк ']
        for i in range(len(codex_names)):
            codex_names[i] = codex_names[i].lower()
        return codex_names

    def fill_requests2(self):   #заполняет класс Request найденной нормой и названием кодекса
        codexes = self.list_of_codexes()
        requests = self.sep_by_requests()
        reg_for_artcicle = ['стать[\w\.]*? [\d\.]*', 'ст\.\s?[\d\.]*']
        reg_for_codex = '\w*? кодекс'
        for r in requests:
            for art in reg_for_artcicle:
                article = re.findall(art, r.answer)
                if article:
                    r.norm.extend(article)
            cod = re.findall(reg_for_codex, r.answer)
            if cod:
                for c in cod:
                    if c.find('настоящ') == -1:
                        r.codex.append(c)
            for a in codexes:
                abbr = re.findall(' ' + a, r.answer)
                if abbr:
                    r.codex.extend(abbr)
            if len(r.codex) >= 1 and len(r.norm) >= 1 and (len(r.codex) == len(r.norm) or len(r.codex) == 1):
                codexes_requests.append(r)



codexes_requests = []


s = Separator("pravoved_articles.txt")
s.fill_requests2()

'''
print(len(codexes_requests))
'''

def dict_codexes_creator():
    codexes = dict()
    codexes[('нк', 'налог')] = [1, 2]
    codexes[('гк', 'граждан')] = [3, 4, 5]
    codexes[('коап', 'администрат')] = [7]
    codexes[('земел', 'зк')] = [8]
    codexes[('тк', 'трудов')] = [9]
    codexes[('жилищ', 'жк')] = [11]
    codexes[('бюджет', 'бк')] = [12]
    codexes[('процессуальн', 'упк')] = [14]
    codexes[('уголов', 'ук')] = [16]
    codexes[('лесн', 'лк')] = [19]
    codexes[('семейн', 'ск')] = [17]
    codexes[('водн', 'вк')] = [20]
    codexes[('гпк')] = [21]
    codexes[('ктм')] = [22]
    codexes[('уик')] = [23]
    codexes[('апк', 'арбитраж')] = [24]
    codexes[('воздушн', 'взк')] = [25]
    codexes[('кввт')] = [26]
    codexes[('тк', 'таможен')] = [912]
    codexes[('грк', 'градостроит')] = [916]
    poss_codexes = ['тк', 'кввт', 'воздушн', 'взк', 'нк', 'налог', 'гк', 'граждан',
                    'коап', 'администрат', 'земел', 'зк', 'тк', 'трудов', 'жилищ', 'жк',
                    'бюджет', 'бк', 'процессульальн', 'упк', 'уголовн', 'ук', 'лесн', 'лк',
                    'семейн', 'ск', 'водн', 'вк', 'гпк', 'ктм', 'уик', 'апк', 'арбитраж']
    return codexes, poss_codexes


def codexes_to_json(codex_directory):
    codexes_out = []
    codexes, poss_codexes = dict_codexes_creator()
    names = list(codexes.keys())
    for cr in codexes_requests:
        new_codex = []
        finded = 0
        if re.search('\d', cr.norm[0]) is None:
            continue
        for cod in cr.codex:
            for n in names:
                if type(n) is tuple:
                    if cod.find(n[0]) != -1 or cod.find(n[1]) != -1:
                        finded = 1
                else:
                    if cod.find(n) != -1:
                        finded = 1
                if finded == 1:
                    new_codex.append(codexes[n])
                    break
        cr.codex = new_codex
        codexes_out.append(cr)

    for co in codexes_out:
        co_n = []
        for norms in co.norm:
            #print(norms)
            co_norm1 = re.search('\d+[\.\d]*', norms)
            if co_norm1:
                co_norm1 = co_norm1[0]
            else:
                continue
            if not co_norm1.endswith('.'):
                co_norm1 += '.'
            co_n.append(co_norm1)
        co.norm = co_n


    set_numbers = set()

    for files in os.listdir(codex_directory):
        file_path = os.path.join(codex_directory, files)
        a, b = coll.iter_pravoved(file_path)
        set_numbers.update(list(a.keys()))

    for co in codexes_out:
        if len(co.codex) == len(co.norm):
            for i in range(len(co.codex)):
                co.cod_norm.append([co.codex[i], co.norm[i]])
        elif len(co.codex) == 1:
            for n in co.norm:
                co.cod_norm.append([*co.codex, n])

    pic = open('pravoved_to_json.json', 'w', encoding='utf-8')
    ans_dict = {}
    for j, co in enumerate(codexes_out):
        new_codnorm = set()
        codnorm = co.cod_norm
        for cn in codnorm:
            cod = cn[0]
            norm = cn[1]
            for c in cod:
                if (str(c), norm) in set_numbers:
                    new_codnorm.add((nc.name_codexes[c].lower(), 'ст ' + norm[:-1]))
        co.cod_norm = list(new_codnorm)
        ans_dict[j] = dict()
        ans_dict[j]["Question"] = co.question
        ans_dict[j]["Answer_Lawyer"] = co.answer
        for i in range(len(co.cod_norm)):
            co.cod_norm[i] = ' '.join(co.cod_norm[i])
        ans_dict[j]["Answer"] = ', '.join(co.cod_norm)
        #json.dump(ans_dict, pic, ensure_ascii=False, indent=2)
        #pic.write('\n\n')
    json.dump(ans_dict, pic, indent=2)

codexes_to_json("codexes")


#norms_codexes_to_normal("codexes")


'''
coll.iter_by_docs()
for co in codexes_out:
    print(co)
print(len(codexes_out))
'''
