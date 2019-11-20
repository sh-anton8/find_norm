# Реализация обратного индекса
import text_transformer as tt
import pickle

# массив документов (законов)
documents_to_dict = tt.Text_Transformer("/Users/shapkin/Downloads/codex_1.txt").to_dict()
documents_keys = documents_to_dict.keys()
# print(documents_keys)
documents = [documents_to_dict[key] for key in documents_keys]
# print(documents[46])

# Запишем все это в файлы, т.к это понадобтися потом
with open('keys.pickle', 'wb') as f:
    pickle.dump(documents_keys, f)

with open('documents.pickle', 'wb') as f:
    pickle.dump(documents, f)

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

# сохраняем обратный индекс на диске
with open('inv_ind.pickle', 'wb') as f:
    pickle.dump(inv_ind, f)