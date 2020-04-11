import os


def make_dir(path_to_dir):
    if not os.path.exists(path_to_dir):
        os.mkdir(path_to_dir)


make_dir("files")
make_dir(os.path.join("files", "tf_idf"))
make_dir(os.path.join("files", "metrics_count"))
make_dir(os.path.join("files", "learning_to_rank"))

