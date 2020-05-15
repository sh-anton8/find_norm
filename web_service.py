from flask import Flask, request, render_template, g
from pi_predict_norm_LTR import predict_norm
from tools.tfidf import TFIDF
from tools.inverse_index import InvIndex

PATH_TO_TFIDF = "files/tf_idf/tf_idf_{}"

app = Flask(__name__)
features = []


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
    answers = predict_norm(text, features)
    print(answers)
    return render_template('upload.html', answers=answers, text=text)


if __name__ == '__main__':
    app.run()
