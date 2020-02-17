import search as search
import tools.pravoved_recognizer as pravoved_recognizer
import tools.tokenize_docs as tokenize_docs
import matplotlib.pyplot as plt
import ml_metrics
from sklearn.metrics import ndcg_score
import math


x = []
y_codex = []
y_articles = []

tfidf_searcher = search.TFIDF_Search(tokenize_docs.Tokenizer('text'), "tfidf(1_1)_codexes_for_article_pickle")
pravoved = pravoved_recognizer.norms_codexes_to_normal("codexes")


def topn_recall():
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


answers = [0] * len(pravoved)
for i in range(len(pravoved)):
    answers[i] = tfidf_searcher.request_processing_input_without_print2(pravoved[i].question)


def map_k():
    actual_cod, predicted_cod = [], []
    actual_art, predicted_art = [], []
    for i in range(1, 100, 2):
        for j in range(len(pravoved)):
            answ = answers[j][:i]
            cod, art = [], []
            for ans in answ:
                cod.append(ans[0][0])
                art.append((ans[0][0], ans[0][2]))
            predicted_cod.append(cod)
            predicted_art.append(art)
            actual_cod.append([pravoved[j].codex])
            actual_art.append([(pravoved[j].codex, pravoved[j].norm)])
        mapk_cod = ml_metrics.mapk(actual_cod, predicted_cod, i)
        mapk_art = ml_metrics.mapk(actual_art, predicted_art, i)
        print(mapk_cod)
        print(mapk_art)
        x.append(i)
        y_codex.append(mapk_cod)
        y_articles.append(mapk_art)

    plt.plot(x, y_codex, color='blue', label='Кодексы')
    plt.plot(x, y_articles, color='red', label='Статьи')
    plt.ylabel('MAP(k)')
    plt.xlabel('Количество статей в топе')
    plt.legend()
    plt.savefig('map.png')
    plt.show()


def ndcg():
    for i in range(2, 101, 2):
        sum_ndcg = 0
        for j in range(len(pravoved)):
            answ = answers[j][:i]
            scores = []
            ndcg = 0
            for ans in answ:
                if (pravoved[j].codex, pravoved[j].norm) == (ans[0][0], ans[0][2]):
                    scores.append(1)
                else:
                    scores.append(0)
            for i in range(1, len(scores) + 1):
                ndcg += scores[i - 1] / math.log(i + 1, 2)
            sum_ndcg += ndcg
        print(sum_ndcg / len(pravoved))
        x.append(i)
        y_articles.append(sum_ndcg / len(pravoved))

    plt.plot(x, y_articles, color='red', label='Статьи')
    plt.ylabel('NDCG')
    plt.xlabel('Количество статей в топе')
    plt.legend()
    plt.savefig('ndcg.png')
    plt.show()


#map_k()
#ndcg()
