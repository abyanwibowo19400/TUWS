import requests
import pandas as pd
import time
import os
from datetime import datetime

# --- Konfigurasi Baru dengan OpenWeatherMap ---
# GANTI DENGAN API KEY ANDA!
API_KEY = "deccdd7ebd0e435d3330bc6165556d1b"
# Koordinat Telkom University
KOTA = "Bandung"
LAT = "-6.974028"
LON = "107.630529"
#PILIH SALAH SATU
#API_URL = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
API_URL= f"https://api.openweathermap.org/data/2.5/weather?q={KOTA}&appid={API_KEY}&units=metric&lang=id"

NAMA_FILE_CSV = "dataset/data_cuaca_historis_bandung.csv"
INTERVAL_DETIK = 60  # 1 menit (1 * 60 detik)

def ambil_dan_simpan_data():
    """
    Fungsi untuk mengambil data cuaca dari OpenWeatherMap API dan menyimpannya.
    """
    try:
        # 1. Mengambil data dari API
        print(f"Mengambil data dari: {API_URL}")
        response = requests.get(API_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # 2. Siapkan data untuk disimpan (struktur JSON OpenWeatherMap berbeda)
        # Menggunakan .get() untuk menghindari error jika data hujan tidak ada
        intensitas_hujan = data.get('rain', {}).get('1h', 0.0)

        data_to_save = {
            'timestamp': pd.to_datetime(datetime.now()),
            'suhu': data['main']['temp'],
            'kelembaban': data['main']['humidity'],
            'kecepatan_angin': data['wind']['speed'],
            'arah_angin': data['wind']['deg'],
            'tekanan_udara': data['main']['pressure'],
            'intensitas_hujan': intensitas_hujan # Intensitas hujan dalam 1 jam terakhir (mm)
        }
        
        df_new = pd.DataFrame([data_to_save])
        
        # 3. Simpan ke file CSV
        if not os.path.exists(NAMA_FILE_CSV):
            df_new.to_csv(NAMA_FILE_CSV, index=False)
            print(f"File {NAMA_FILE_CSV} dibuat. Data pertama disimpan.")
        else:
            df_new.to_csv(NAMA_FILE_CSV, mode='a', header=False, index=False)
            print(f"Data baru ditambahkan pada {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error saat mengambil data dari API: {e}")
    except KeyError as e:
        print(f"Error: Key tidak ditemukan dalam respons JSON. Mungkin format data berubah. Key: {e}")
        print("Raw Response:", data)
    except Exception as e:
        print(f"Terjadi error tak terduga: {e}")

# --- Loop Utama untuk Menjalankan Skrip ---
if __name__ == "__main__":
    if API_KEY == "GANTI_DENGAN_API_KEY_ANDA":
        print("!!! PENTING: Harap ganti 'GANTI_DENGAN_API_KEY_ANDA' dengan API Key OpenWeatherMap Anda di dalam skrip.")
    else:
        print("Memulai skrip pengumpul data cuaca menggunakan OpenWeatherMap...")
        while True:
            ambil_dan_simpan_data()
            print(f"Menunggu {INTERVAL_DETIK / 60} menit untuk pengambilan data selanjutnya...")
            time.sleep(INTERVAL_DETIK)