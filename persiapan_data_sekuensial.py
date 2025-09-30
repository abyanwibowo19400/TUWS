import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- KONFIGURASI ---
FILE_INPUT = "dataset/data_cuaca_historis_bandung.csv"
FILE_OUTPUT_X = "X_sekuensial.npy"
FILE_OUTPUT_Y = "y_multi_output.npy"

# Konfigurasi Sekuens
LANGKAH_WAKTU_INPUT = 30  # Gunakan data 30 menit terakhir sebagai input
LANGKAH_WAKTU_OUTPUT = [5, 10, 15, 20] # Prediksi untuk 5, 10, 15, 20 menit ke depan

# ==============================================================================
# FUNGSI BANTU
# ==============================================================================
def tentukan_kelas_prediksi(intensitas_hujan, arah_angin):
    if intensitas_hujan <= 0:
        return 0
    if 337.5 <= arah_angin < 360 or 0 <= arah_angin < 22.5: return 1
    if 22.5 <= arah_angin < 67.5: return 2
    if 67.5 <= arah_angin < 112.5: return 3
    if 112.5 <= arah_angin < 157.5: return 4
    if 157.5 <= arah_angin < 202.5: return 5
    if 202.5 <= arah_angin < 247.5: return 6
    if 247.5 <= arah_angin < 292.5: return 7
    return 8

# ==============================================================================
# PROGRAM UTAMA
# ==============================================================================
if __name__ == "__main__":
    print("--- Memulai Persiapan Data Sekuensial ---")
    
    # 1. Muat dan siapkan data awal
    df = pd.read_csv(FILE_INPUT)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Data asli dimuat: {len(df)} baris.")

    fitur = ['suhu', 'kelembaban', 'kecepatan_angin', 'arah_angin', 'tekanan_udara', 'intensitas_hujan']
    
    # 2. Scaling Fitur (Penting untuk Deep Learning)
    scaler = StandardScaler()
    df[fitur] = scaler.fit_transform(df[fitur])
    print("Scaling fitur selesai.")

    # 3. Buat Target Awal (untuk setiap baris)
    # Ini diperlukan untuk membuat target multi-output nanti
    df['target_kelas'] = df.apply(lambda row: tentukan_kelas_prediksi(row['intensitas_hujan'], row['arah_angin']), axis=1)

    # 4. Buat dataset sekuensial
    print("Membuat dataset sekuensial...")
    X_sekuensial, y_multi_output = [], []
    max_output_step = max(LANGKAH_WAKTU_OUTPUT)

    # Loop dari awal hingga akhir data yang memungkinkan
    for i in range(LANGKAH_WAKTU_INPUT, len(df) - max_output_step):
        # Input: data dari 30 menit sebelumnya
        input_seq = df.loc[i - LANGKAH_WAKTU_INPUT : i - 1, fitur].values
        X_sekuensial.append(input_seq)
        
        # Output: prediksi untuk 5, 10, 15, 20 menit ke depan
        output_targets = [df.loc[i + step - 1, 'target_kelas'] for step in LANGKAH_WAKTU_OUTPUT]
        y_multi_output.append(output_targets)

    X_final = np.array(X_sekuensial)
    y_final = np.array(y_multi_output)
    
    print(f"Dataset sekuensial berhasil dibuat.")
    print(f"Bentuk data X: {X_final.shape}") # (jumlah_sampel, 30 langkah, 6 fitur)
    print(f"Bentuk data y: {y_final.shape}") # (jumlah_sampel, 4 target)

    # 5. Simpan data yang sudah diproses
    np.save(FILE_OUTPUT_X, X_final)
    np.save(FILE_OUTPUT_Y, y_final)
    print(f"\nData yang telah diproses disimpan sebagai '{FILE_OUTPUT_X}' dan '{FILE_OUTPUT_Y}'")
    
    # Simpan juga scaler agar bisa digunakan saat prediksi live
    import joblib
    joblib.dump(scaler, 'scaler.joblib')
    print("Scaler juga telah disimpan sebagai 'scaler.joblib'")