# Реализация обратного индекса
import tokenizer
import pickle
from collections import Counter

# запрос
request = tokenizer.Tokenizer("Я получил возврат денег в размере 5000 руб. " +
                              "с государственного сайта, надо ли мне платить с этого возрата налог?").tokenize()
print(request)

with open('inv_ind.pickle', 'rb') as f:
    inv_ind = pickle.load(f)

with open('keys.pickle', 'rb') as f:
    documents_keys = pickle.load(f)


# склеивание из массива в строку (на будущее для TF-IDF и n-gramm)
def glue_func(array):
    ans = array[0]
    for i in range(1, len(array)):
        ans += ' ' + array[i]
    return ans


# поиск документов по запросу
# возвращает Counter, где каждой статье в соответсвие ставится количество слов, которые встречаются в ней из запроса
# inv_ind - инвертированный индекс, заранее преобработанный и записанный на файле
def find2(request):
    ans = Counter()
    for word in request:
        if word in inv_ind:
            for el in inv_ind[word]:
                ans[el] += 1
    return ans


ans = find2(request)
#print(ans)


def get_ans(ans, documents_keys):
    top = ans.most_common(5)
    i = 1
    for el in top:
        print(i, '-ый по релевантности ответ: ', list(documents_keys)[el[0]], sep='')
        i += 1


get_ans(ans, documents_keys)