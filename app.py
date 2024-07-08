from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Muat model
model = joblib.load('model/svm.pkl')
label_encoders = joblib.load('model/label_encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictions', methods=["POST"])
def predictions():
    # storage_location = label_encoders.keys()
    date = pd.to_datetime(41701.0, origin='1899-12-30', unit='D')
    
    # Ambil data dari form HTML
    storage_bin = float(request.form['storage_bin'])
    storage_location = float(request.form['storage_location'])
    gr_date_str = request.form['gr_date']
    gr_date = (pd.to_datetime(gr_date_str) - pd.to_datetime('1899-12-30')).days
    total_stock = float(request.form['total_stock'])
    last_movement = float(request.form['last_movement'])
    
    # Lakukan prediksi menggunakan model (contoh dummy)
    print([storage_bin, storage_location, gr_date, total_stock, last_movement])
    input_data = np.array([[storage_bin, storage_location, gr_date, total_stock, last_movement]])

    predict = model.predict(input_data)
    print(gr_date)
    
    # Kembalikan hasil prediksi sebagai respons JSON
    return jsonify({'predict': predict.tolist()})

if __name__ == '__main__':
    app.run(debug=True)