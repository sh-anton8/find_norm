import re

class Request:
    def __init__(self, theme, question, answer):
        self.theme = theme
        self.question = question
        self.answer = answer
        self.norm = []
        self.codex = []

    def __str__(self):
        return 'answer:{}\n\nnorm: {}\ncodex: {}\n-------------\n'\
            .format(self.answer, self.norm, self.codex)



class Separator:
    def __init__(self, path_to_file):
        self.path_to_file = path_to_file

    def editor(self):                   #деление текста на запросы
        file = open(self.path_to_file, 'r')
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
        codex_names = ['УИК ', 'КАС', 'СК ', 'ЖК ', 'АПК ', 'УПК ', 'ГПК ', 'ук ', 'НК ', 'ТК ', 'ГК ', 'КоАП']
        for i in range(len(codex_names)):
            codex_names[i] = codex_names[i].lower()
        return codex_names

    def fill_requests(self):   #заполняет класс Request найденной нормой и названием кодекса
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
                r.codex.extend(cod)
            for a in codexes:
                if r.answer.find(a) != -1:
                    r.codex.append(a)
            print(r)


'''
s = Separator("pravoved_articles.txt")
s.fill_requests()
'''
