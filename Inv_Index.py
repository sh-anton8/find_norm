# Реализация обратного индекса
import text_transformer as tt
import tokenizer
from collections import Counter


# массив документов (законов)
documents_to_dict = tt.Text_Transformer("/Users/shapkin/Downloads/codex_1.txt").to_dict()
documents_keys = documents_to_dict.keys()
documents = [documents_to_dict[key] for key in documents_keys]
print(documents[0])

# запрос
request = tokenizer.Tokenizer("Я получил возврат денег в размере 5000 руб. " +
                              "с государственного сайта, надо ли мне платить с этого возврата налог?").tokenize()
print(request)


# токенизация по пробелу своими руками
# однако предобработка отдает уже стокенизированные данные => функция не нужна
def tokenization(docs):
    tokenized_docs = []
    for i in range(len(docs)):
        doc_tokens = []
        for word in docs[i].split():
            doc_tokens.append(word)
        tokenized_docs.append(doc_tokens)
    return tokenized_docs


# инвертированный (обратный) индекс своими руками
def inverse_index(docs):
    inv_ind = {}
    for i in range(len(docs)):
        for token in docs[i][1]:
            if token in inv_ind:
                if i not in inv_ind[token]:
                    inv_ind[token].append(i)
            else:
                inv_ind[token] = [i]
    return inv_ind


# пример работы
inv_ind = inverse_index(documents)
print(inv_ind)


def get_docs_for_word(word, inv_ind):
    ans = 0
    if (word in inv_ind):
        for num in inv_ind[word]:
            ans += (2 ** num)
    return ans


# поиск документов по запросу
def find(request, inv_ind):
    docs_of_words = []
    for word in request:
        docs_of_words.append(get_docs_for_word(word, inv_ind))
    if len(request) > 1:
        ans = docs_of_words[0] & docs_of_words[1]
        for i in range(2, len(request)):
            ans &= docs_of_words[i]
    else:
        ans = docs_of_words[0]
    ans_arr = []
    ind = 0
    while (ans > 0):
        if (ans % 2 == 1):
            ans_arr.append(ind)
        ind += 1
        ans //= 2
    return ans_arr


# поиск документов по запросу ВАРИАНТ 2
def find2(request, inv_ind):
    ans = Counter()
    for word in request:
        if word in inv_ind:
            for el in inv_ind[word]:
                ans[el] += 1
    return ans

print(find2(request, inv_ind))
