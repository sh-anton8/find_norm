import search as search
import tools.pravoved_recognizer as pravoved_recognizer
import tools.tokenize_docs as tokenize_docs
import matplotlib.pyplot as plt


x = []
y_codex = []
y_articles = []

tfidf_searcher = search.TFIDF_Search(tokenize_docs.Tokenizer('text'), "files/tfidf(1_1)_codexes_for_article_pickle")
pravoved = pravoved_recognizer.norms_codexes_to_normal("codexes")

for i in range(1, 101, 2):
    codex = 0
    article = 0
    both = 0
    for pr in pravoved:
        cod = 0
        art = 0
        answers = tfidf_searcher.request_processing_input_without_print(pr.question, i)
        for ans in answers:
            if ans[0][0] == pr.codex:
                cod = 1
            if ans[0][2] == pr.norm:
                art = 1
        if cod == 1 and art == 1:
            both += 1
        elif cod == 1:
            codex += 1
        elif article == 1:
            art += 1

    x.append(i)
    y_codex.append(codex + both)
    y_articles.append(both)

plt.plot(x, y_codex, color='blue', label='Кодексы')
plt.plot(x, y_articles, color='red', label='Статьи')
plt.title('Точность в зависимости от количества статей в топе')
plt.ylabel('Попаданий')
plt.xlabel('Количество статей в топе')
plt.legend()
plt.show()