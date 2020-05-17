# Антон

import os
import tools.coll as coll
import pickle
from tqdm import tqdm
import pandas as pd
from collections import Counter
from tqdm import tqdm


class InvIndex:
    def __init__(self, tokenizer):
        self.inv_ind = {}
        self.doc_ids = []
        self.doc_lens = {}
        self.tokenizer = tokenizer
    
    
    def build_on(self, corpus, tokenized=True):
        """
        Построить обратный индекс по корпусу.
        tokenized: если True, то считаем что подается уже токенизированный
        корпус (тем же токенайзером что задан в init)
        """
        for doc_id, doc_text in corpus:
            doc_num = len(self.doc_ids)
            self.doc_ids.append(doc_id)

            tokens = doc_text if tokenized else self.tokenizer.tokenize(doc_text)
            tokens = set(tokens)  # важно - не учитываем повторы токенов
            
            self.doc_lens[doc_id] = len(tokens)

            for token in tokens:
                if token not in self.inv_ind:
                    self.inv_ind[token] = []
                # в обратный индекс записываем не сам doc_id, а его номер,
                # это делается для оптимизации памяти и скорости
                self.inv_ind[token].append(doc_num)


    def search(self, query, topN=10, threshold=0.5, metric='recall'):
        """
        Поиск по запросу.
        topN - сколько документов оставить после ранжирования по релевантности
        threshold - порог отсечения по релевантности
        metric - тип релевантности: точность, полнота или F1
        """
        assert metric in ['precision', 'recall', 'f_measure']
        
        tokens = set(self.tokenizer.tokenize(query))
        qlen = len(tokens)

        # заполняем счетчик количества токенов запроса в данном "документе" корпуса
        query_tokens_counter = Counter()
        for token in tokens:
            doc_num_list = self.inv_ind.get(token, [])
            query_tokens_counter.update(doc_num_list)

        result = []
        for doc_num in query_tokens_counter:
            found = query_tokens_counter[doc_num]

            recall = found / qlen
            precision = found / self.doc_lens[self.doc_ids[doc_num]]

            if precision + recall != 0.0:
                f_measure = 2 * precision * recall / (precision + recall)
            else:
                f_measure = 0.0
            
            if metric == 'recall':
                rel = recall
            elif metric == 'precision':
                rel = precision
            elif metric == 'f_measure':
                rel = f_measure

            if rel < threshold:
                continue

            result.append([self.doc_ids[doc_num], rel])

        if result:
            df_res = pd.DataFrame(result, columns=['doc_id', 'rel'])
            df_res = df_res.sort_values('rel', ascending=False)

            # в результатах поиска может быть много документов  с минимальной
            # релевантностью, но надо их не потерять и взять все, а не только topN
            min_rel_topN = df_res['rel'][:topN].min()
            df_res = df_res[df_res['rel'] >= min_rel_topN]
            result = df_res.values.tolist()
            
        return result


    def save(self, file):
        print('Saving index to: {}'.format(file))
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file):
        print('Loading index from: {}'.format(file))
        with open(file, 'rb') as f:
            return pickle.load(f)



class InvertIndexForHighlight(InvIndex):
    # corpus не токенизирован ранее
    def __init__(self, tokenizer, corpus):
        InvIndex.__init__(self, tokenizer)
        self.token_to_words = {}
        self.word_to_token = {}
        self.corpus = corpus

    def build(self):
        t = tqdm(total=len(self.corpus))
        for doc_id, doc_text in self.corpus:
            norm_tokens = self.tokenizer.tokenize(doc_text)
            tokens = self.tokenizer.tokenize_without_normalizing(doc_text)
            for i in range(len(norm_tokens)):
                if (self.token_to_words.get(norm_tokens[i]) == None):
                    self.token_to_words[norm_tokens[i]] = [tokens[i]]
                else:
                    self.token_to_words[norm_tokens[i]].append(tokens[i])

                if (self.word_to_token.get(tokens[i]) == None):
                    self.word_to_token[tokens[i]] = norm_tokens[i]
            t.update(1)
        t.close()

    # answer - номер ответа в нашей нумерации
    def hightlight_words(self, query, answer):
        tokens = self.tokenizer.tokenize(query)

        doc = self.corpus.get_doc(answer)
        doc_tokens = self.tokenizer.tokenize_without_normalizing(doc)

        bit_card = [0] * len(doc_tokens)
        for ind in range(len(doc_tokens)):
            if self.word_to_token[doc_tokens[ind]] in tokens:
                for num in range(ind - 3, ind + 4):
                    if num >= 0 and num < len(doc_tokens):
                        if bit_card[num] != 2:
                            bit_card[num] = 1
                bit_card[ind] = 2

        print_flag = True

        highlight_words = []
        for ind in range(len(doc_tokens)):
            if bit_card[ind] == 1:
                highlight_words.append(doc_tokens[ind])
                print_flag = True
            elif bit_card[ind] == 2:
                jj = open("file.txt", 'a')
                jj.write('<b>' + doc_tokens[ind] + '</b>')
                highlight_words.append('<b>' + doc_tokens[ind] + '</b>')
                print_flag = True
            elif print_flag:
                highlight_words.append("...")
                print_flag = False
        return ' '.join(highlight_words)

    def save(self, file):
        print('Saving InvertIndexForHighlight to: {}'.format(file))
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file):
        print('Loading InvertIndexForHighlight from: {}'.format(file))
        with open(file, 'rb') as f:
            return pickle.load(f)










