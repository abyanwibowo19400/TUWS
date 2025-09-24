import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- KONFIGURASI ---
FILE_INPUT_X = "X_sekuensial.npy"
FILE_INPUT_Y = "y_multi_output.npy"
FILE_OUTPUT_MODEL = "model_lstm_cuaca.h5" # Model Keras disimpan dalam format .h5

# ==============================================================================
# 1. MEMUAT DATA YANG SUDAH DIPROSES
# ==============================================================================
print("--- Langkah 1: Memuat Data Sekuensial ---")
try:
    X = np.load(FILE_INPUT_X)
    y = np.load(FILE_INPUT_Y)
except FileNotFoundError:
    print("Error: Pastikan file 'X_sekuensial.npy' dan 'y_multi_output.npy' sudah ada.")
    print("Jalankan skrip 'persiapan_data_sekuensial.py' terlebih dahulu.")
    exit()

print(f"Data berhasil dimuat.")
print(f"Bentuk data X: {X.shape}")
print(f"Bentuk data y: {y.shape}")

# Ambil informasi bentuk data untuk arsitektur model
JUMLAH_SAMPEL, LANGKAH_WAKTU_INPUT, JUMLAH_FITUR = X.shape
JUMLAH_OUTPUT_STEPS = y.shape[1] # Jumlah langkah waktu yang diprediksi (misal: 4)
JUMLAH_KELAS = 9 # 0 (Tidak Hujan) + 8 Arah Angin

# ==============================================================================
# 2. MEMBANGUN ARSITEKTUR MODEL LSTM
# ==============================================================================
print("\n--- Langkah 2: Membangun Arsitektur Model LSTM ---")
model = Sequential()
# Layer LSTM pertama dengan return_sequences=True agar bisa ditumpuk
model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(LANGKAH_WAKTU_INPUT, JUMLAH_FITUR)))
model.add(Dropout(0.2)) # Dropout untuk mencegah overfitting
# Layer LSTM kedua
model.add(LSTM(32, activation='relu'))
model.add(Dropout(0.2))
# Layer Dense tersembunyi
model.add(Dense(32, activation='relu'))
# Layer Output: Ini adalah bagian multi-output
# Kita perlu mengubah bentuk output agar sesuai dengan (4 langkah prediksi * 9 kelas)
model.add(Dense(JUMLAH_OUTPUT_STEPS * JUMLAH_KELAS, activation='softmax'))
# Mengubah bentuk output akhir menjadi (batch_size, 4 langkah, 9 kelas)
model.add(tf.keras.layers.Reshape((JUMLAH_OUTPUT_STEPS, JUMLAH_KELAS)))


# Compile model
# 'sparse_categorical_crossentropy' cocok karena target y kita adalah angka integer (0-8)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Tampilkan ringkasan arsitektur model
model.summary()

# ==============================================================================
# 3. MEMBAGI DATA & MELATIH MODEL
# ==============================================================================
print("\n--- Langkah 3: Membagi Data dan Melatih Model ---")
# Bagi data menjadi 80% training dan 20% validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Ukuran data training: {len(X_train)}")
print(f"Ukuran data validasi: {len(X_val)}")

# Latih model
# epochs=20: Model akan melihat seluruh data training sebanyak 20 kali
# batch_size=64: Model akan memproses 64 sampel sekaligus dalam satu iterasi
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_val, y_val)
)

# ==============================================================================
# 4. VISUALISASI HASIL TRAINING (OPSIONAL TAPI BERGUNA)
# ==============================================================================
print("\n--- Langkah 4: Visualisasi Hasil Training ---")
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Loss Training')
plt.plot(history.history['val_loss'], label='Loss Validasi')
plt.title('Grafik Loss Model Selama Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('grafik_loss_model.png')
plt.show()
# ... (setelah blok kode VISUALISASI HASIL TRAINING) ...

# ==============================================================================
# 5. EVALUASI DETAIL PADA DATA VALIDASI
# ==============================================================================
print("\n--- Langkah 5: Evaluasi Detail Model ---")
from sklearn.metrics import classification_report

# Lakukan prediksi pada data validasi
y_pred_raw = model.predict(X_val)
# Ubah hasil probabilitas menjadi kelas prediksi (0-8)
y_pred = np.argmax(y_pred_raw, axis=2)

# Definisikan nama kelas untuk laporan
nama_kelas = [
    'Tidak Hujan', 'Hujan dari Utara', 'Hujan dari Timur Laut',
    'Hujan dari Timur', 'Hujan dari Tenggara', 'Hujan dari Selatan',
    'Hujan dari Barat Daya', 'Hujan dari Barat', 'Hujan dari Barat Laut'
]

# Karena ini adalah model multi-output, mari kita evaluasi setiap langkah prediksi
# y_val -> (jumlah_sampel, 4 langkah prediksi)
# y_pred -> (jumlah_sampel, 4 langkah prediksi)
for i, langkah in enumerate([5, 10, 15, 20]):
    print(f"\n===== Laporan Klasifikasi untuk Prediksi +{langkah} Menit =====")
    # Ambil semua prediksi dan target untuk langkah waktu ke-i
    print(classification_report(
        y_val[:, i], 
        y_pred[:, i],
        labels=range(9),
        target_names=nama_kelas,
        zero_division=0
    ))


# ==============================================================================
# 6. MENYIMPAN MODEL
# ==============================================================================

model.save(FILE_OUTPUT_MODEL)
print(f"Model telah disimpan sebagai '{FILE_OUTPUT_MODEL}'")