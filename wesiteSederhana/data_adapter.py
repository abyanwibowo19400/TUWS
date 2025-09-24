# data_adapter.py

import requests
import pandas as pd

# --- Fungsi-Fungsi Konversi Unit ---
def fahrenheit_to_celsius(f):
    return (f - 32) * 5.0/9.0

def mph_to_mps(mph):
    return mph * 0.44704

def inhg_to_hpa(inhg):
    return inhg * 33.8639

def inch_to_mm(inch):
    return inch * 25.4
# data_adapter.py

# ... (fungsi-fungsi konversi unit Anda sebelumnya) ...

def derajat_ke_mata_angin(derajat):
    """Mengubah derajat arah angin menjadi label teks mata angin."""
    if 337.5 <= derajat < 360 or 0 <= derajat < 22.5:
        return "Utara"
    elif 22.5 <= derajat < 67.5:
        return "Timur Laut"
    elif 67.5 <= derajat < 112.5:
        return "Timur"
    elif 112.5 <= derajat < 157.5:
        return "Tenggara"
    elif 157.5 <= derajat < 202.5:
        return "Selatan"
    elif 202.5 <= derajat < 247.5:
        return "Barat Daya"
    elif 247.5 <= derajat < 292.5:
        return "Barat"
    else: # 292.5 <= derajat < 337.5
        return "Barat Laut"

# ... (sisa file data_adapter.py) ...

# --- Fungsi Adaptor Utama ---
def dapatkan_data_live_untuk_prediksi(url):
    """
    Mengambil data dari API Weather.com, melakukan konversi unit,
    dan mengembalikannya dalam format DataFrame yang siap pakai.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        obs = data['observations'][0]
        # Ambil arah angin dalam derajat
        arah_angin_derajat = obs['winddir']
        arah_angin_teks = derajat_ke_mata_angin(arah_angin_derajat)
        
        # Lakukan semua konversi unit
        suhu_c = fahrenheit_to_celsius(obs['imperial']['temp'])
        kecepatan_angin_mps = mph_to_mps(obs['imperial']['windSpeed'])
        tekanan_hpa = inhg_to_hpa(obs['imperial']['pressure'])
        intensitas_hujan_mm = inch_to_mm(obs['imperial']['precipRate'])
        
        
        # Susun data ke dalam DataFrame dengan nama kolom yang konsisten
        data_final_df = pd.DataFrame([{
            'suhu': suhu_c,
            'kelembaban': obs['humidity'],
            'kecepatan_angin': kecepatan_angin_mps,
            'arah_angin': arah_angin_derajat,
            'tekanan_udara': tekanan_hpa,
            'intensitas_hujan': intensitas_hujan_mm
        }])
        
        return data_final_df, obs['obsTimeLocal'], arah_angin_teks

    except requests.exceptions.RequestException as e:
        print(f"Error saat mengambil data dari API: {e}")
        return None, None
    except (KeyError, IndexError) as e:
        print(f"Error: Struktur data JSON dari API tidak sesuai. {e}")
        return None, None