import re
import tokenizer


class Text_Transformer:
    def __init__(self, file_name):
        self.file_name = file_name
        self.file = ''
        self.splt = []

    def delete_garb(self):
        self.file = open(self.file_name, 'r')
        self.file = self.file.read()
        self.file = re.sub("\(в ред\..*?\)", "", self.file)
        self.file = re.sub("\(абзац введен.*?\)", "", self.file)
        self.file = re.sub("r'\d\d\.\d\d\.\d{4}", "", self.file)
        self.file = self.file.replace("(см. текст в предыдущей редакции)", "")


    def split_text(self, type_word, word="", finder=""):
        self.splt = []
        if type_word == 'Глава':
            cur = finder.find(word)
            prev = finder.find(word)
            while cur != -1:
                cur = finder.find(word, prev + 1)
                self.splt.append(finder[prev:cur])
                prev = cur
            if cur == -1 and prev == 0:
                return None
            self.splt.append(finder[prev:])
        elif type_word == "Статья":
            self.splt = []
            prev = finder.find(word)
            cur = prev
            while cur != -1:
                cur = finder.find(word, prev + 1)
                split_name = re.search("Статья .*?[А-Я]", finder[prev - 1:cur])[0][:-1]
                self.splt.append([finder[prev:cur], split_name])
                prev = cur
        elif type_word == "Пункт":
            self.splt = re.split("\n\d\. ", finder)
        elif type_word == "Кодекс":
            self.splt = re.split("------------------------------------------------------------------\n", finder)
        return self.splt

    def to_dict(self):
        self.delete_garb()
        codex_id = 0
        all_codes = {}
        for codex in self.split_text("Кодекс", "", s.file):
            codex_name = re.search(".*?\n", codex)[0]
            for chap in self.split_text("Глава", "Глава ", codex):
                for art in self.split_text("Статья", "Статья ", chap):
                    par_num = 1
                    for par in self.split_text("Пункт", "", art[0]):
                        key_name = str(codex_name + art[1] + "Пункт " + str(par_num)).replace("\n", " ")
                        par_num += 1
                        t = tokenizer.Tokenizer(par)
                        all_codes[key_name] = [codex_id, t.tokenize()]
                        codex_id += 1
        return all_codes



s = Text_Transformer("/Users/anastasia/Downloads/codex_1.txt")
print(s.to_dict().keys())



#k = s.split_text("Глава", "Глава ")
#f = s.split_text("Статья", "Статья ", k[1])
#print(f[0])
#for k in enumerate(s.split_text("Пункт", "", f[0][0])):
#    print(k)
#for v in s:
 #   k = split_text2(v, "Статья ")
  #  for l in k:
   #     l[0] = l[0].replace(l[1], '')
    #    print(l[1])
     #   #print(to_normal.tokenize(l[0]))
      #  print(l[0])
