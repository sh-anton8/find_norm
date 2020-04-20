import os
from tools.pravoved_recognizer import norms_codexes_to_normal


# создание необходимых директорий
os.makedirs("files", exist_ok=True)
os.makedirs(os.path.join("files", "tf_idf"), exist_ok=True)
os.makedirs(os.path.join("files", "metrics_count"), exist_ok=True)
os.makedirs(os.path.join("files", "learning_to_rank"), exist_ok=True)
os.makedirs(os.path.join("files", "features_df"), exist_ok=True)

# создание файла json с запросами правоведа
norms_codexes_to_normal("codexes", save_to_json=True)
