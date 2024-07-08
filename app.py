from flask import Flask, render_template
import joblib

app = Flask(__name__)

# Muat model
model = joblib.load('model/svm.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictions')
def predictions():
    return null

if __name__ == '__main__':
    app.run(debug=True)