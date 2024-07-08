from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import joblib
import csv

app = Flask(__name__)

# Muat model
model = joblib.load('model/svm.pkl')
label_encoders = joblib.load('model/label_encoders.pkl')

@app.route('/')
def home():
    select = []
    with open('model/stg_bin.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            select.append(row[0])
    return render_template('index.html', data=select)

@app.route('/select')
def select():
    select = []
    with open('model/stg_bin.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            select.append(row[0])
    return select

@app.route('/predictions', methods=["POST"])
def predictions():
    date = pd.to_datetime(41701.0, origin='1899-12-30', unit='D')

    last_movement_str = request.form['last_movement']
    gr_date_str = request.form['gr_date']
    
    # Ambil data dari form HTML
    storage_bin = float(request.form['storage_bin'])
    storage_location = float(request.form['storage_location'])
    total_stock = float(request.form['total_stock'])
    gr_date = (pd.to_datetime(gr_date_str) - pd.to_datetime('1899-12-30')).days
    last_movement = (pd.to_datetime(last_movement_str) - pd.to_datetime('1899-12-30')).days
    
    input_data = np.array([[storage_bin, storage_location, gr_date, total_stock, last_movement]])

    predict = model.predict(input_data)

    # Kembalikan hasil prediksi sebagai respons JSON
    return jsonify({'predict': predict.tolist()[0]})

if __name__ == '__main__':
    app.run(debug=True)