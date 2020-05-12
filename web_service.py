from flask import Flask, request, render_template
from x2_build_index import predict_norm


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = 'zip'

app = Flask(__name__)


@app.route('/', methods=['GET'])
def upload_file():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def main_page():
    text = request.form['text']
    answers = predict_norm(text)
    print(answers)
    return render_template('upload.html', answers=answers, text=text)


if __name__ == '__main__':
    app.run()
