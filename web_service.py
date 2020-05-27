from flask import Flask, request, render_template, g
from pi_9_predict_norm_LTR import predict_norm
from tools.tfidf import TFIDF
from tools.inverse_index import InvIndex, InvertIndexForHighlight

import os
from tools.relative_paths_to_directories import path_to_directories
PATH_TO_TFIDF = "files/tf_idf/tf_idf_{}"
PATH_TO_ROOT, PATH_TO_TOOLS, PATH_TO_FILES, PATH_TO_TF_IDF, PATH_TO_INV_IND, PATH_TO_BM_25, \
    PATH_TO_LEARNING_TO_RANK = path_to_directories(os.getcwd())

from flask import Markup

Markup('<p> flkfjlj </p>').unescape()

app = Flask(__name__)
features = []
iifh = InvertIndexForHighlight.load(os.path.join(PATH_TO_FILES, "IIHighlight"))


@app.before_first_request
def load_tfidf():
    for i in range(1, 7):
        features.append(TFIDF.load(PATH_TO_TFIDF.format(i)))
    features.append(InvIndex.load("files/inv_ind.pickle"))


@app.route('/', methods=['GET'])
def upload_file():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def main_page():
    text = request.form['text']
    answers = predict_norm(text, features, iifh)
    print(answers)
    return render_template('upload.html', answers=answers, text=text)


if __name__ == '__main__':
    app.run()
