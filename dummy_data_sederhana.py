import pandas as pd
import numpy as np

# --- KONFIGURASI ---
FILE_INPUT = "dataset/data_cuaca_historis_bandung.csv"
FILE_OUTPUT = "dataset/data_cuaca_augmented.csv" # Nama file baru untuk data yang sudah ditambah
JUMLAH_TEMPLATE = 20 # Jumlah data hujan terakhir yang dijadikan template 10 - 30

# ==============================================================================
# FUNGSI BANTU UNTUK MENENTUKAN LABEL ARAH ANGIN
# ==============================================================================
def tentukan_arah(arah_angin):
    """Mengubah derajat arah angin menjadi label teks (Utara, Timur Laut, dll.)"""
    if 337.5 <= arah_angin < 360 or 0 <= arah_angin < 22.5:
        return "Utara"
    elif 22.5 <= arah_angin < 67.5:
        return "Timur Laut"
    elif 67.5 <= arah_angin < 112.5:
        return "Timur"
    elif 112.5 <= arah_angin < 157.5:
        return "Tenggara"
    elif 157.5 <= arah_angin < 202.5:
        return "Selatan"
    elif 202.5 <= arah_angin < 247.5:
        return "Barat Daya"
    elif 247.5 <= arah_angin < 292.5:
        return "Barat"
    else: # 292.5 <= arah_angin < 337.5
        return "Barat Laut"

# ==============================================================================
# PROGRAM UTAMA
# ==============================================================================
if __name__ == "__main__":
    print("--- Memulai Proses Augmentasi Data Cuaca ---")
    
    # 1. Baca data asli
    try:
        df = pd.read_csv(FILE_INPUT)
        print(f"Berhasil memuat {len(df)} baris dari '{FILE_INPUT}'")
    except FileNotFoundError:
        print(f"ERROR: File '{FILE_INPUT}' tidak ditemukan. Pastikan nama file sudah benar.")
        exit()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 2. Saring data untuk menemukan semua momen saat hujan terjadi
    df_hujan = df[df['intensitas_hujan'] > 0].copy()

    if df_hujan.empty:
        print("Tidak ditemukan data hujan di dalam file. Proses dihentikan.")
        exit()

    # 3. Deteksi arah angin mana yang datanya belum ada SAAT HUJAN
    df_hujan['label_arah'] = df_hujan['arah_angin'].apply(tentukan_arah)
    semua_arah = {"Utara", "Timur Laut", "Timur", "Tenggara", "Selatan", "Barat Daya", "Barat", "Barat Laut"}
    arah_yang_ada = set(df_hujan['label_arah'].unique())
    arah_yang_hilang = semua_arah - arah_yang_ada
    
    df_final = df.copy()

    if arah_yang_hilang:
        print(f"\nTerdeteksi arah hujan yang datanya hilang: {sorted(list(arah_yang_hilang))}")
        
        template_data = df_hujan.tail(JUMLAH_TEMPLATE)
        dummy_rows = []
        
        # Peta untuk mengubah nama arah kembali ke rentang derajat
        arah_map_derajat = {
            "Utara": (337.5, 22.5), "Timur Laut": (22.5, 67.5), "Timur": (67.5, 112.5),
            "Tenggara": (112.5, 157.5), "Selatan": (157.5, 202.5), "Barat Daya": (202.5, 247.5),
            "Barat": (247.5, 292.5), "Barat Laut": (292.5, 337.5)
        }
        
        print("Membuat data dummy dengan penyesuaian musiman...")
        for arah in sorted(list(arah_yang_hilang)):
            arah_min, arah_max = arah_map_derajat[arah]

            for _, row in template_data.iterrows():
                new_row = row.copy()
                
                # Logika khusus untuk arah Utara yang melintasi 360/0 derajat
                if arah == "Utara":
                    if np.random.rand() < 0.5:
                        new_row['arah_angin'] = np.random.uniform(337.5, 360.0)
                    else:
                        new_row['arah_angin'] = np.random.uniform(0.0, 22.5)
                else:
                    new_row['arah_angin'] = np.random.uniform(arah_min, arah_max)

                # Penyesuaian ilmiah untuk meniru musim hujan
                new_row['kelembaban'] = min(100, row['kelembaban'] + np.random.uniform(5, 10))
                new_row['suhu'] = row['suhu'] - np.random.uniform(0.5, 1.5)
                new_row['tekanan_udara'] -= np.random.uniform(0.5, 1.0)
                new_row['intensitas_hujan'] = max(row['intensitas_hujan'], np.random.uniform(0.5, 3.0))
                
                # Hapus kolom label_arah yang hanya untuk bantuan
                if 'label_arah' in new_row:
                    del new_row['label_arah']
                
                dummy_rows.append(new_row)
        
        if dummy_rows:
            df_dummy = pd.DataFrame(dummy_rows)
            df_final = pd.concat([df, df_dummy], ignore_index=True)
            print(f"Berhasil membuat dan menambahkan {len(df_dummy)} baris data dummy.")

    else:
        print("\nSemua arah angin saat hujan sudah terwakili. Tidak perlu membuat data dummy.")

    # 4. Simpan hasilnya ke file baru
    df_final.to_csv(FILE_OUTPUT, index=False)
    print(f"\nProses selesai. Data gabungan telah disimpan di: '{FILE_OUTPUT}'")
    print(f"Jumlah baris data akhir: {len(df_final)}")