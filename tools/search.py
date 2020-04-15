# Антон

from tools.tfidf import TFIDF
import os
import numpy as np

class Baseline_Search():
    def __init__(self, tf_idf_path, tokenizer):
        self.tokenizer = tokenizer
        self.tf1 = TFIDF.load(tf_idf_path + '_' + str(1))
        self.tf2 = TFIDF.load(tf_idf_path + '_' + str(2))
        self.tf3 = TFIDF.load(tf_idf_path + '_' + str(3))
        self.tf4 = TFIDF.load(tf_idf_path + '_' + str(4))
        self.tf5 = TFIDF.load(tf_idf_path + '_' + str(5))
        self.tf6 = TFIDF.load(tf_idf_path + '_' + str(6))

    def search(self, query, topN = 10, threshold=0.5):
        tokens = self.tokenizer.tokenize(query)
        sim1 = self.tf1.similarity(tokens)
        sim2 = self.tf2.similarity(tokens)
        sim3 = self.tf3.similarity(tokens)
        sim4 = self.tf4.similarity(tokens)
        sim5 = self.tf5.similarity(tokens)
        sim6 = self.tf6.similarity(tokens)
        sim1 += sim2 + sim3 + sim4 + sim5 + sim6
        sim1 /= 6
        sims_indices = (-sim1).argsort()[:topN]
        sims = [(self.tf1.doc_ids[si], sim1[si]) for si in sims_indices \
                if sim1[si] >= threshold]
        return sims
