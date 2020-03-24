from tools.inverse_index import InvIndex
from tools.bm25 import My_BM
from collections import Counter
import os

class Features():
    def __init__(self, inv_ind_pickle, bm_25_file):
        self.inv_ind = InvIndex.load(inv_ind_pickle)
        if not os.path.isfile(bm_25_file):
            self.bm_obj = My_BM(inv_ind_pickle)
            self.bm_obj.save(bm_25_file)
        else:
            self.bm_obj = My_BM.load(bm_25_file)

    def get_fmerRelev_feature(self, req_text, article_num):
        reqst = req_text
        self.inv_ind.tokenizer.text = reqst
        reqst = self.inv_ind.tokenizer.tokenize(self.inv_ind.cash, self.inv_ind.morph, self.inv_ind.stop_words)
        help = Counter()
        ans = dict()
        for word in reqst:
            if word in self.inv_ind.inv_ind:
                for key in self.inv_ind.inv_ind[word]:
                    help[key[0]] += 1
        Recall = help[article_num] / len(reqst)
        if (len(self.inv_ind.num_tokens_dict[article_num]) == 0):
            F = -1000000000
        else:
            Presicion = help[article_num] / len(self.inv_ind.num_tokens_dict[article_num])
            if (help[article_num] == 0):
                F = -1000000000
            else:
                F = 2 / ((1 / Presicion) + (1 / Recall))
        return F

    def get_doc_len_feature(self, req_text, article_num):
        return self.inv_ind.num_to_len[article_num]


    def get_bm25_feature(self, req_text, article_num):
        return self.bm_obj.get_feature(req_text, article_num)


