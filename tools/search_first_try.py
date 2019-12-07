import coll
import numpy as np
import pickle
import tokenizer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


def glue_func(array):
    ans = array[0]
    for i in range(1, len(array)):
        ans += ' ' + array[i]
    return ans


class Search:
    def __init__(self, documents):
        #self.analyzer = analyzer
        #self.vectorizer = vectorizr
        file = open(documents, 'r')
        f = file.read()
        self.collection = coll.Collection(f)
        self.key_to_text, self.text_to_key = coll.iter_by_par(self.collection)
        self.inv_ind = {}
        self.counted_tfidf = {}

    def inversed_index(self):
        for doc in coll.par(self.collection):
            tokens = tokenizer.Tokenizer(doc).tokenize()
            for token in tokens:
                if token in self.inv_ind:
                    if self.text_to_key[doc] not in self.inv_ind[token]:
                        self.inv_ind[token].append(self.text_to_key[doc])
                else:
                    self.inv_ind[token] = [self.text_to_key[doc]]
        return self.inv_ind

    def save_index(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self.inv_ind, f)

    def take_index_from_file(self, file):
        with open(file, 'rb') as f:
            self.inv_ind = pickle.load(f)
        return self.inv_ind

    def request_processing(self, request):
        reqst = tokenizer.Tokenizer(request).tokenize()
        ans = Counter()
        for word in reqst:
            if word in self.inv_ind:
                for el in self.inv_ind[word]:
                    ans[el] += 1
        top = ans.most_common(5)
        i = 1
        for el in top:
            print(i, '-ый по релевантности ответ: ', el[0], sep='')
            i += 1

    def tfidf_counting(self, documents):
        docs = [glue_func(text[1]) for text in documents]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(docs)
        X_T = np.array(X.todense()).transpose()
        names = vectorizer.get_feature_names()
        for i in range(len(names)):
            self.counted_tfidf[names[i]] = []
            for j in range(len(docs)):
                self.counted_tfidf[names[i]].append(X_T[i][j])


# file = open("codex_1.txt", 'r')
# f = file.read()
# colle = coll.Collection(f)
# J, Z = coll.iter_by_par(colle)
# print(Z[J[list(J.keys())[100]]])

# for k in coll.par(colle):
#    print(k)
#    print('---')

srch = Search("codex_1.txt")
srch.inversed_index()
srch.request_processing("Я получил возврат денег в размере 5000 руб. с государственного сайта, надо ли мне платить с этого возрата налог?")

















