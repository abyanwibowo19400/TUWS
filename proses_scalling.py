import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# --- 1. MUAT DATA ANDA ---
try:
    df = pd.read_csv("dataset/data_cuaca_historis_bandung.csv", encoding='latin1')
except FileNotFoundError:
    print("Pastikan file 'data_cuaca_augmented.csv' ada di folder yang sama.")
    exit()

# Definisikan fitur dan target (contoh target biner sederhana)
fitur = ['suhu', 'kelembaban', 'kecepatan_angin', 'arah_angin', 'tekanan_udara', 'intensitas_hujan']
target = (df['intensitas_hujan'] > 0).astype(int) # Ganti dengan target multi-kelas Anda jika perlu

X = df[fitur]
y = target

# --- 2. BAGI DATA MENJADI TRAINING DAN TESTING (SANGAT PENTING!) ---
# Scaling HARUS dilakukan SETELAH membagi data untuk mencegah "kebocoran data"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- DATA SEBELUM SCALING ---")
print(X_train.describe())

# --- 3. BUAT DAN TERAPKAN SCALER ---
# Buat objek StandardScaler
scaler = StandardScaler()

# "Pelajari" parameter (mean & std) dari data training DAN transformasikan
X_train_scaled = scaler.fit_transform(X_train)

# Terapkan transformasi yang SAMA pada data testing
# Kita hanya menggunakan .transform() di sini, bukan .fit_transform()
X_test_scaled = scaler.transform(X_test)

print("\n--- DATA SETELAH SCALING ---")
# Ubah kembali ke DataFrame agar mudah dibaca
df_scaled_train = pd.DataFrame(X_train_scaled, columns=fitur)
print(df_scaled_train.describe())

# --- 4. SIMPAN SCALER UNTUK DIGUNAKAN NANTI ---
# Scaler ini penting untuk memproses data baru di aplikasi web Anda
nama_file_scaler = 'scalerData.joblib'
joblib.dump(scaler, nama_file_scaler)
print(f"\nScaler telah disimpan sebagai '{nama_file_scaler}'")