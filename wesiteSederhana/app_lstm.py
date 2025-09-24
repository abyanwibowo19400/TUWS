import pandas as pd
import numpy as np
import requests
import joblib
from keras.models import load_model
from flask import Flask, render_template

# --- KONFIGURASI ---
API_KEY = "deccdd7ebd0e435d3330bc6165556d1b"
KOTA = "Bandung"
URL = f"https://api.openweathermap.org/data/2.5/weather?q={KOTA}&appid={API_KEY}&units=metric&lang=id"

# Konfigurasi yang harus SAMA PERSIS dengan saat persiapan data
LANGKAH_WAKTU_INPUT = 30
LANGKAH_WAKTU_OUTPUT = [5, 10, 15, 20]
FITUR = ['suhu', 'kelembaban', 'kecepatan_angin', 'arah_angin', 'tekanan_udara', 'intensitas_hujan']

# --- INISIALISASI APLIKASI DAN MODEL ---
app = Flask(__name__)

# Muat model dan scaler
print("--- Memuat model dan scaler ---")
try:
    model = load_model('model_lstm_cuaca.h5')
    scaler = joblib.load('scaler.joblib')
    print("Model dan scaler berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat model atau scaler: {e}")
    model, scaler = None, None

# Label nama kelas untuk menerjemahkan hasil prediksi
LABEL_MAP = {
    0: 'Cerah / Berawan', 1: 'Hujan dari Utara', 2: 'Hujan dari Timur Laut',
    3: 'Hujan dari Timur', 4: 'Hujan dari Tenggara', 5: 'Hujan dari Selatan',
    6: 'Hujan dari Barat Daya', 7: 'Hujan dari Barat', 8: 'Hujan dari Barat Laut'
}

# ==============================================================================
# HALAMAN UTAMA WEBSITE
# ==============================================================================
@app.route('/')
def halaman_utama():
    if model is None or scaler is None:
        return "Error: Model atau scaler tidak berhasil dimuat. Aplikasi tidak bisa berjalan."

    try:
        # --- SIMULASI PENGAMBILAN DATA 30 MENIT TERAKHIR ---
        # Di dunia nyata, Anda akan mengambil ini dari database live.
        # Untuk sekarang, kita simulasikan dengan membaca data historis terakhir.
        df_historis = pd.read_csv("data_cuaca_augmented.csv", encoding='latin1')
        data_terkini_sequence = df_historis.tail(LANGKAH_WAKTU_INPUT)
        
        if len(data_terkini_sequence) < LANGKAH_WAKTU_INPUT:
            return f"Tidak cukup data historis (butuh {LANGKAH_WAKTU_INPUT}, hanya ada {len(data_terkini_sequence)})."

        # Ambil data paling baru untuk ditampilkan sebagai "cuaca saat ini"
        cuaca_saat_ini = data_terkini_sequence.iloc[-1]
        
        # --- PROSES DATA UNTUK PREDIKSI ---
        # 1. Scaling
        data_scaled = scaler.transform(data_terkini_sequence[FITUR])
        # 2. Ubah format menjadi 3D: (1 sampel, 30 langkah waktu, 6 fitur)
        input_untuk_model = np.expand_dims(data_scaled, axis=0)
        
        # --- LAKUKAN PREDIKSI ---
        prediksi_raw = model.predict(input_untuk_model)
        # Hasilnya akan berbentuk (1, 4, 9). Kita ambil kelas dengan probabilitas tertinggi.
        prediksi_angka = np.argmax(prediksi_raw, axis=2)[0]
        
        # --- TERJEMAHKAN HASIL PREDIKSI ---
        hasil_prediksi = []
        for i, langkah in enumerate(LANGKAH_WAKTU_OUTPUT):
            prediksi_teks = LABEL_MAP.get(prediksi_angka[i], "Tidak Diketahui")
            hasil_prediksi.append({'waktu': langkah, 'prediksi': prediksi_teks})

        # --- KIRIM DATA KE FRONTEND ---
        return render_template('index_lstm.html', 
                               suhu=f"{cuaca_saat_ini['suhu']:.1f}",
                               kelembapan=cuaca_saat_ini['kelembaban'],
                               deskripsi="-", # Deskripsi dari API tidak digunakan di sini
                               hasil_prediksi=hasil_prediksi)
    
    except Exception as e:
        return f"Terjadi error saat proses: {e}"

# ==============================================================================
# FILE HTML (ditempatkan di dalam skrip agar mudah)
# ==============================================================================
@app.route('/templates/<path:filename>')
def serve_template(filename):
    return render_template(filename)

# Menjalankan aplikasi
if __name__ == '__main__':
    # Buat file HTML dummy jika belum ada
    html_content = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <title>Prediksi Cuaca LSTM</title>
    <style>
        body { font-family: sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #f0f4f8; }
        .card { background: #fff; padding: 2rem; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); text-align: center; width: 400px; }
        .current-weather { margin-bottom: 2rem; border-bottom: 1px solid #eee; padding-bottom: 1rem; }
        .prediction-grid { display: grid; grid-template-columns: 1fr 2fr; gap: 0.5rem; text-align: left; }
        .prediction-grid strong { font-weight: bold; }
    </style>
</head>
<body>
    <div class="card">
        <h2>Prediksi Cuaca Multi-Output LSTM</h2>
        <div class="current-weather">
            <h3>Cuaca Saat Ini</h3>
            <p>Suhu: <strong>{{ suhu }}°C</strong> | Kelembapan: <strong>{{ kelembapan }}%</strong></p>
        </div>
        <div class="predictions">
            <h3>Hasil Prediksi</h3>
            <div class="prediction-grid">
                {% for p in hasil_prediksi %}
                    <strong>+ {{ p.waktu }} menit:</strong>
                    <span>{{ p.prediksi }}</span>
                {% endfor %}
            </div>
        </div>
    </div>
</body>
</html>
"""
    with open("templates/index_lstm.html", "w") as f:
        f.write(html_content)
        
    app.run(debug=True)