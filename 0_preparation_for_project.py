import os

'''
Создание всех необходимых папок для дальнейшего проекта
'''

os.makedirs("files", exist_ok=True)
os.makedirs(os.path.join("files", "tf_idf"), exist_ok=True)
os.makedirs(os.path.join("files", "metrics_count"), exist_ok=True)
os.makedirs(os.path.join("files", "learning_to_rank"), exist_ok=True)

