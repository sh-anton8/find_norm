# Реализация обратного индекса
import tokenizer
import pickle

# запрос
request = tokenizer.Tokenizer("Я получил возврат денег в размере 5000 руб. " +
                              "с государственного сайта, надо ли мне платить с этого возрата налог?").tokenize()
print(request)

with open('inv_ind.pickle', 'rb') as f:
    inv_ind = pickle.load(f)

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

# данный поиск выдает все файлы, в которых содержатся все слова из запроса и ломается иначе
# поиск основан на работе с масками (как с подмножествами)
