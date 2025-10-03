import pandas as pd
import os
import glob

# --- KONFIGURASI ---
FOLDER_DATA_MENTAH = "data_mentah_harian"
FILE_OUTPUT_MASTER = "dataset/data_cuaca_telkom_MASTER.csv"

# --- PETA POSISI KOLOM ---
# Kita definisikan posisi setiap data yang kita butuhkan dan nama barunya
# Format: {indeks_kolom: 'nama_baru'}
KOLOM_INDEX_MAP = {
    0: 'timestamp',
    1: 'suhu',
    4: 'kelembaban',
    20: 'kecepatan_angin',
    22: 'arah_angin',
    24: 'tekanan_udara',
    12: 'intensitas_hujan',
    10: 'intensitas_cahaya'
}

def proses_file_excel(nama_file):
    """Membaca file Excel berdasarkan posisi kolom, membersihkannya, dan mengembalikan DataFrame."""
    try:
        # Baca file Excel, lewati 2 baris header pertama, dan jangan gunakan header
        df = pd.read_excel(nama_file, header=None, skiprows=2)
        
        # Ambil hanya kolom yang kita butuhkan berdasarkan posisinya (indeks)
        indeks_kolom_yang_diambil = list(KOLOM_INDEX_MAP.keys())
        df_terpilih = df[indeks_kolom_yang_diambil].copy()
        
        # Ganti nama kolom dari indeks menjadi nama yang kita inginkan
        df_terpilih.rename(columns=KOLOM_INDEX_MAP, inplace=True)
        
        # Proses Pembersihan (tetap sama)
        df_bersih = df_terpilih
        df_bersih['timestamp'] = pd.to_datetime(df_bersih['timestamp'])
        for col in df_bersih.columns:
            if col != 'timestamp':
                df_bersih[col] = pd.to_numeric(df_bersih[col], errors='coerce')
        
        df_bersih.dropna(subset=['timestamp'], inplace=True) # Hapus baris jika timestamp kosong
        df_bersih.fillna(method='ffill', inplace=True)
        df_bersih.fillna(0, inplace=True)
        
        return df_bersih
    except Exception as e:
        print(f"Gagal memproses file {nama_file}: {e}")
        return None

if __name__ == "__main__":
    print("--- Memulai proses penggabungan data harian (strategi indeks kolom) ---")
    
    pola_pencarian = os.path.join(FOLDER_DATA_MENTAH, '*.xlsx')
    semua_file = glob.glob(pola_pencarian)
    
    if not semua_file:
        print(f"Tidak ada file .xlsx ditemukan di folder '{FOLDER_DATA_MENTAH}'.")
        exit()
        
    print(f"Ditemukan {len(semua_file)} file untuk diproses.")
    
    list_df = []
    for file in semua_file:
        df_cleaned = proses_file_excel(file)
        if df_cleaned is not None:
            list_df.append(df_cleaned)
            
    if not list_df:
        print("Tidak ada file yang berhasil diproses. Proses dihentikan.")
        exit()
    
    df_master = pd.concat(list_df, ignore_index=True)
    
    df_master.sort_values('timestamp', inplace=True)
    df_master.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
    
    df_master.to_csv(FILE_OUTPUT_MASTER, index=False)
    
    print(f"\n--- Proses Selesai! ---")
    print(f"'{FILE_OUTPUT_MASTER}' telah berhasil dibuat/diperbarui.")
    print(f"Total baris data yang digabungkan: {len(df_master)}")
    print(f"Data dari {df_master['timestamp'].min()} hingga {df_master['timestamp'].max()}")