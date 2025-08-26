import streamlit as st
import numpy as np
import joblib

# --- FUNGSI UNTUK MEMUAT MODEL DAN SCALER ---
# @st.cache_resource digunakan agar model dan scaler hanya dimuat sekali
@st.cache_resource
def load_model_and_scaler():
    """Memuat model regresi dan scaler yang sudah dilatih."""
    try:
        scaler = joblib.load('scaler.joblib')
        model = joblib.load('model_regresi.joblib')
        return scaler, model
    except FileNotFoundError:
        st.error("File model atau scaler tidak ditemukan. Pastikan file 'scaler.joblib' dan 'model_regresi.joblib' ada di folder yang sama.")
        return None, None

# --- MEMUAT MODEL DAN SCALER ---
scaler, model = load_model_and_scaler()

# --- TAMPILAN ANTARMUKA PENGGUNA (UI) DENGAN STREAMLIT ---
st.title('Prediksi Harga Rumah di California 🏠')
st.write('Aplikasi ini memprediksi median harga rumah di California berdasarkan beberapa fitur.')

# Membuat sidebar untuk input dari pengguna
st.sidebar.header('Masukkan Fitur Rumah:')

def user_input_features():
    """Membuat sidebar untuk input fitur dari pengguna."""
    med_inc = st.sidebar.slider('Pendapatan Median (MedInc)', 0.5, 15.0, 3.87, 0.1)
    house_age = st.sidebar.slider('Umur Rumah (HouseAge)', 1.0, 52.0, 28.0, 1.0)
    avg_rooms = st.sidebar.slider('Rata-rata Kamar (AveRooms)', 1.0, 20.0, 5.4, 0.1)
    avg_bedrms = st.sidebar.slider('Rata-rata Kamar Tidur (AveBedrms)', 0.5, 10.0, 1.1, 0.1)
    population = st.sidebar.slider('Populasi (Population)', 200.0, 38000.0, 1425.0, 100.0)
    avg_occup = st.sidebar.slider('Rata-rata Penghuni (AveOccup)', 1.0, 20.0, 3.0, 0.1)
    latitude = st.sidebar.slider('Latitude', 32.5, 42.0, 35.6, 0.1)
    longitude = st.sidebar.slider('Longitude', -124.5, -114.0, -119.5, 0.1)
    
    # Kumpulkan data input ke dalam dictionary
    data = {
        'MedInc': med_inc,
        'HouseAge': house_age,
        'AveRooms': avg_rooms,
        'AveBedrms': avg_bedrms,
        'Population': population,
        'AveOccup': avg_occup,
        'Latitude': latitude,
        'Longitude': longitude
    }
    return data

# Ambil input dari pengguna
input_data = user_input_features()

# Tombol untuk membuat prediksi
if st.sidebar.button('Prediksi Harga'):
    if model is not None and scaler is not None:
        # Ubah dictionary input menjadi numpy array 2D
        input_array = np.array(list(input_data.values())).reshape(1, -1)
        
        # Lakukan scaling pada data input
        input_scaled = scaler.transform(input_array)
        
        # Lakukan prediksi dengan model
        prediction = model.predict(input_scaled)
        
        # Tampilkan hasil prediksi
        st.subheader('Hasil Prediksi:')
        # Ingat, target kita dalam satuan $100,000
        predicted_price = prediction[0] * 100000
        st.success(f'Prediksi median harga rumah adalah: ${predicted_price:,.2f}')
    else:
        st.warning("Model tidak dapat dimuat. Prediksi tidak dapat dilakukan.")

st.write("---")
st.write("Dibuat sebagai bagian dari proses belajar Machine Learning End-to-End.")