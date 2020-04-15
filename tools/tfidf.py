# Антон

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class TFIDF:
    def __init__(self, tokenizer, vectorizer_params=None):
        self.tokenizer = tokenizer
        self.doc_ids = []
        if vectorizer_params:
            self.vectorizer = TfidfVectorizer(**vectorizer_params)
        else:
            self.vectorizer = TfidfVectorizer()
        
    def build_on(self, corpus, tokenized=True):
        """
        Построить обратный индекс по корпусу.
        tokenized: если True, то считаем что подается уже токенизированный
        корпус (тем же токенайзером что задан в init)
        """

        # для sklearn.TfidfVectorizer приходится подавать токенизированный
        # корпус как склеенные строки токенов.
        # в gensim проще, можно сразу подавать списки токенов
        joined_corpus = []
        for doc_id, doc_text in corpus:
            self.doc_ids.append(doc_id)

            tokens = doc_text if tokenized else self.tokenizer.tokenize(doc_text)
            joined_corpus.append(" ".join(tokens))
        self.tfidf_matrix = self.vectorizer.fit_transform(joined_corpus)

    def search(self, query, topN=10, threshold=0.5):
        """
        Поиск по запросу.
        topN - сколько документов оставить после ранжирования по релевантности
        threshold - порог отсечения по релевантности.
        Релевантность здесь - косинусная мера между вектором запроса и документа.
        """
        tokens = self.tokenizer.tokenize(query)
       
        query_tfidf = self.vectorizer.transform([" ".join(tokens)])
        raw_sims = cosine_similarity(query_tfidf, self.tfidf_matrix).reshape(-1)
        sims_indices = (-raw_sims).argsort()[:topN]
        # вернуть результат как список [(doc_id, rel), ...]
        sims = [(self.doc_ids[si], raw_sims[si]) for si in sims_indices \
               if raw_sims[si] >= threshold]
        return sims

    def similarity(self, query_tokens):
        query_tfidf = self.vectorizer.transform([" ".join(query_tokens)])
        raw_sims = cosine_similarity(query_tfidf, self.tfidf_matrix).reshape(-1)
        return raw_sims

    def save(self, file):
        print('Saving tf-idf to: {}'.format(file))
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file):
        print('Loading tfidf from: {}'.format(file))
        with open(file, 'rb') as f:
            return pickle.load(f)
