# find_norm
Репозиторий проекта "Поиск нормы права по ситуации".

В codexes хранятся все кодексы. \
В files хранятся все файлы, необходимые для проекта:\
files/learning_to_rank -- файлы, где записаны признаки 
и разделение на группы тестовой и тренировочной выборок при обучении ранжированию, также предсказания модели. \
files/metrics_count --  Посчитанные метрики для модели \
files/tf_idf --  Посчитанные tf-idf \
files/bm_25.pickle -- Посчитанный bm_25 \
files/inv_ind.pickle -- Посчитанный обратный индекс
files/pravoved_articles.txt -- Ответы на запросы Правоведа \
\
\
В папке tools расположены все вспомогательные модули \
В основной директории хранятся файлы для запуска программы \
0_preparation_for_project.py -- Создание нужных директорий для дальнейшей работы проекта \
2_build_index.py -- Построение обратного индекса, если он еще не построен, и его сохранение \
3_build_tfidf.py --  Построение tf-idf, если он еще не построен, и его сохранение \
4_predict_norm.py -- Поиск, основанный либо на tf-idf, либо на обратном индексе \
5_bm_preprocessing.py -- Построение bm-25 \
6_learning_to_rank.py -- Построение модели обучения ранжированию и предсказания результатов \
7_metrics_for_xgboost.py - Расчет метрик качества ранжирования для предсказаний \ 


Для построения модели нужно запускать все файлы в корневой директории по порядку, ПРЕДВАРИТЕЛЬНО! обозначив TRAIN_SAMPLE, TEST_SAMPLE в 6_learning_to_rank.py, 7_metrics_for_xgboost.py.\
Рекомендованный TRAIN_SAMPLE = 1200, TEST_SAMPLE = 1429
