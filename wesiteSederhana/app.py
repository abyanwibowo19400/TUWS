# import requests
# import joblib
# import pandas as pd
# from flask import Flask, render_template

# # Inisialisasi aplikasi Flask
# app = Flask(__name__)

# # --- Konfigurasi ---
# API_KEY = "deccdd7ebd0e435d3330bc6165556d1b"
# KOTA = "Bandung"
# URL = f"https://api.weather.com/v2/pws/observations/current?stationId=IBANDU75&format=json&units=e&apiKey=88509dac19764a48909dac19763a482f"

# # --- Muat Model Machine Learning ---
# # Model ini dimuat hanya sekali saat aplikasi pertama kali dijalankan
# print("Memuat model machine learning...")
# try:
#     # model = joblib.load('model_prediksi_hujan.joblib')
#     model = joblib.load('model_prediksi_hujan_darimana_XGBoost.joblib')
#     print("Model berhasil dimuat.")
# except FileNotFoundError:
#     print("Error: File model 'model_prediksi_hujan._darimanajoblib' tidak ditemukan.")
#     print("Pastikan Anda sudah menjalankan 'latih_model.py' untuk membuat file model.")
#     model = None

# label_map = {
#     0: 'Cerah / Berawan',
#     1: 'Berpotensi Hujan dari Arah Utara',
#     2: 'Berpotensi Hujan dari Arah Timur Laut', 
#     3: 'Berpotensi Hujan dari Arah Timur',
#     4: 'Berpotensi Hujan dari Arah Tenggara',
#     5: 'Berpotensi Hujan dari Arah Selatan',
#     6: 'Berpotensi Hujan dari Arah Barat Daya',
#     7: 'Berpotensi Hujan dari Arah Barat',
#     8: 'Berpotensi Hujan dari Arah Barat Laut'
# }
# # Membuat halaman utama website
# @app.route('/')
# def halaman_utama():
#     if model is None:
#         return "Error: Model tidak berhasil dimuat. Aplikasi tidak bisa berjalan."

#     try:
#         # 1. Ambil data cuaca terkini dari API
#         response = requests.get(URL)
#         response.raise_for_status()
#         data = response.json()
        
#         # 2. Ekstrak informasi untuk ditampilkan (seperti sebelumnya)
#         suhu = data['main']['temp']
#         kelembapan = data['main']['humidity']
#         deskripsi_cuaca = data['weather'][0]['description'].title()
#         nama_kota = data['name']
#         kode_ikon = data['weather'][0]['icon']
#         url_ikon = f"https://openweathermap.org/img/wn/{kode_ikon}@2x.png"

#         # 3. Siapkan data untuk input prediksi model
#         # Strukturnya harus sama persis seperti saat training!
#         data_untuk_prediksi = pd.DataFrame([{
#             'suhu': suhu,
#             'kelembaban': kelembapan,
#             'kecepatan_angin': data['wind']['speed'],
#             'arah_angin': data['wind']['deg'],
#             'tekanan_udara': data['main']['pressure'],
#             # Gunakan nilai hujan dari API, atau 0 jika tidak ada data hujan
#             'intensitas_hujan': data.get('rain', {}).get('1h', 0.0) 
#         }])
        
#         # 4. Lakukan Prediksi
#         # Lakukan Prediksi, hasilnya akan berupa angka (misal: [2])
#         hasil_prediksi_angka = model.predict(data_untuk_prediksi)[0]
#         hasil_prediksi_teks = label_map.get(hasil_prediksi_angka, "Tidak Diketahui")
    
#         # 5. Kirim semua data (data terkini + hasil prediksi) ke HTML
#         return render_template('index.html', 
#                                kota=nama_kota, 
#                                suhu=f"{suhu:.1f}",
#                                kelembapan=kelembapan,
#                                deskripsi=deskripsi_cuaca, 
#                                ikon=url_ikon,
#                                prediksi=hasil_prediksi_teks) # <- Kirim hasil prediksi
    
#     except Exception as e:
#         return f"Terjadi error saat proses: {e}"

# # Menjalankan aplikasi
# if __name__ == '__main__':
#     app.run(debug=True)
# app.py

# app.py

import joblib
import pandas as pd
from flask import Flask, render_template
# Impor fungsi dari file terpisah yang sudah kita buat
from data_adapter import dapatkan_data_live_untuk_prediksi

# --- Konfigurasi ---
# URL sekarang menggunakan API Weather.com yang baru
URL = "https://api.weather.com/v2/pws/observations/current?stationId=IBANDU75&format=json&units=e&apiKey=88509dac19764a48909dac19763a482f"

# --- Muat Model Machine Learning ---
print("Memuat model machine learning...")
try:
    # Ganti dengan nama file model XGBoost Anda
    model = joblib.load('model_prediksi_hujan_darimana_XGBoost.joblib')
    print("Model berhasil dimuat.")
except FileNotFoundError:
    print("Error: File model tidak ditemukan.")
    model = None

# --- Inisialisasi Aplikasi Flask ---
app = Flask(__name__)

label_map = {
    0: 'Cerah / Berawan', 1: 'Berpotensi Hujan dari Arah Utara',
    2: 'Berpotensi Hujan dari Arah Timur Laut', 3: 'Berpotensi Hujan dari Arah Timur',
    4: 'Berpotensi Hujan dari Arah Tenggara', 5: 'Berpotensi Hujan dari Arah Selatan',
    6: 'Berpotensi Hujan dari Arah Barat Daya', 7: 'Berpotensi Hujan dari Arah Barat',
    8: 'Berpotensi Hujan dari Arah Barat Laut'
}

# Halaman utama website
@app.route('/')
def halaman_utama():
    if model is None:
        return "Error: Model tidak berhasil dimuat."

    try:
        # 1. Panggil fungsi dari file lain untuk mendapatkan data yang sudah bersih & terkonversi
        data_untuk_prediksi, waktu_observasi, arah_angin_teks = dapatkan_data_live_untuk_prediksi(URL)
        
        if data_untuk_prediksi is None:
            return "Gagal mendapatkan atau memproses data dari API live."

        # 2. Lakukan Prediksi
        hasil_prediksi_angka = model.predict(data_untuk_prediksi)[0]
        hasil_prediksi_teks = label_map.get(hasil_prediksi_angka, "Tidak Diketahui")
    
        # Ambil data yang sudah dikonversi untuk ditampilkan
        data_tampil = data_untuk_prediksi.iloc[0]

        # 3. Kirim semua data ke HTML
        return render_template('index.html', 
                               kota="Stasiun IBANDU75", 
                               suhu=f"{data_tampil['suhu']:.1f}",
                               kelembapan=data_tampil['kelembaban'],
                               deskripsi=f"Data diambil pada: {waktu_observasi}", 
                               ikon="https://openweathermap.org/img/wn/10d@2x.png", # Ikon statis sebagai contoh
                               prediksi=hasil_prediksi_teks,
                               kecepatan_angin =f"{data_tampil['kecepatan_angin']:.2f}",
                               arah_angin=arah_angin_teks)
    
    except Exception as e:
        return f"Terjadi error saat proses: {e}"

# Menjalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)