import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# --- 1. MEMUAT DAN MEMPERSIAPKAN DATA ---
print("--- Memuat dan Mempersiapkan Data ---")
# Ganti dengan nama file data lengkap Anda jika berbeda
df = pd.read_csv("dataset/data_cuaca_historis_bandung.csv", encoding='latin1')

# Lakukan feature engineering target seperti sebelumnya
# (Ini disederhanakan, Anda bisa copy-paste fungsi dari skrip sebelumnya jika perlu)
df['target_prediksi'] = (df['intensitas_hujan'] > 0).astype(int) # Contoh sederhana

fitur = ['suhu', 'kelembaban', 'kecepatan_angin', 'arah_angin', 'tekanan_udara', 'intensitas_hujan']
X = df[fitur]
y = df['target_prediksi']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling data (wajib untuk SVM)
# scaler = StandardScaler()
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)
try:
    # scaler = joblib.load('scalerData.joblib')
    scaler = joblib.load('scalerData.joblib')
except FileNotFoundError:
    print("Sclaer tidak ditemukan")
    exit()

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 2. DEFINISIKAN GRID PARAMETER UNTUK DICOBA ---
print("\n--- Mendefinisikan Grid Parameter ---")
# Ini adalah daftar "tombol putar" yang akan dicoba oleh GridSearchCV
# Kita mulai dengan rentang yang luas
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

# --- 3. INISIALISASI DAN JALANKAN GRIDSEARCHCV ---
print("\n--- Memulai Proses GridSearchCV (Mungkin akan memakan waktu lama) ---")
# cv=3: Menggunakan 3-fold cross-validation
# n_jobs=-1: Menggunakan semua core CPU agar lebih cepat
# verbose=2: Menampilkan progres
grid_search = GridSearchCV(
    estimator=SVC(random_state=42, class_weight='balanced'),
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2
)

# Latih grid search pada data
grid_search.fit(X_train_scaled, y_train)

# --- 4. TAMPILKAN HASIL TERBAIK ---
print("\n--- Proses Selesai! ---")
print("Parameter terbaik yang ditemukan:")
print(grid_search.best_params_)

print("\nAkurasi cross-validation terbaik:")
print(f"{grid_search.best_score_:.2%}")

# --- 5. EVALUASI MODEL TERBAIK PADA DATA TESTING ---
print("\n--- Mengevaluasi model terbaik pada data testing ---")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Simpan model terbaik yang sudah di-tuning
joblib.dump(best_model, 'model_svm_tuned.joblib')
print("\nModel SVM terbaik telah disimpan sebagai 'model_svm_tuned.joblib'")