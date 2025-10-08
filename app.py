# app.py - Aplikasi Streamlit Transliterasi Aksara Jawa
import streamlit as st
import os
import tempfile
import joblib
from PIL import Image
import backend  # Import semua fungsi dari backend.py
from huggingface_hub import hf_hub_download

# ============================================================
# CSS untuk Tampilan Beige dan Harmonis (Kontras Lebih Baik)
# ============================================================

# Warna Latar Belakang Utama: #F5F5DC (Beige/Krem)
# Warna Teks/Judul Gelap: #654321 (Dark Brown - Kontras Terbaik)
# Warna Aksen/Tombol: #A0522D (Sienna - Cokelat Kemerahan Lembut)

st.markdown(
    """
    <style>
    /* Mengubah Latar Belakang Utama Aplikasi menjadi Beige */
    .stApp {
        background-color: #F5F5DC; 
        color: #654321; /* Warna teks default diubah ke Dark Brown */
    }
    
    /* Mengubah warna teks judul utama dan garis pemisah */
    .stApp h1 {
        color: #654321 !important; /* Dark Brown untuk judul */
        text-align: center;
        border-bottom: 2px solid #A0522D; /* Sienna untuk garis bawah */
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    /* Mengubah warna teks subjudul/heading */
    .stApp h4, .stApp h3 {
        color: #A0522D !important; /* Sienna untuk subjudul */
        border-left: 5px solid #654321; /* Dark Brown untuk garis tepi */
        padding-left: 10px;
        margin-top: 20px;
    }
    
    /* Styling Teks Biasa dan Label - Kontras Tinggi */
    .stApp label, .stApp p, .stApp span {
        color: #4B4B4B; /* Cokelat gelap untuk keterbacaan */
    }
    
    /* Tombol Browse File pada st.file_uploader */
    div[data-testid="stFileUploader"] button {
        color: white !important;
        background-color: #A0522D !important;
    }

    /* Styling Info Box - Lebih Lembut dan Teks Gelap */
    .stApp div[data-testid="stInfo"] {
        background-color: #FEF9E7; 
        border-left: 5px solid #A0522D; 
        border-radius: 8px;
        padding: 10px;
        color: #4B4B4B; /* Teks harus gelap */
    }
    
    /* Styling Warning Box - Lebih Lembut dan Teks Gelap */
    .stApp div[data-testid="stWarning"] {
        background-color: #FCF3CF; 
        border-left: 5px solid #B8860B; 
        border-radius: 8px;
        padding: 10px;
        color: #4B4B4B; /* Teks harus gelap */
    }
    
    /* Styling Success Box - Hapus warna hijau yang mengganggu, ganti dengan cokelat */
    .stApp div[data-testid="stSuccess"] {
        background-color: #E9F7EF; /* Tetap sedikit hijau pucat, tapi teks gelap */
        border-left: 5px solid #A0522D; 
        border-radius: 8px;
        padding: 10px;
        color: #4B4B4B; /* Teks harus gelap */
    }

    /* Styling Text Area dan Input */
    .stApp textarea, .stApp input {
        border: 1px solid #A0522D !important;
        border-radius: 8px;
        background-color: white; 
        color: #4B4B4B !important;
    }
    
    /* Styling Text Area Konten (hasil transliterasi) */
    div[data-testid="stTextarea"] label {
        color: #A0522D !important; /* Warna label Text Area */
    }

    /* Styling Footer */
    .footer-style {
        text-align: center; 
        color: #654321; /* Dark Brown */
        padding-top: 10px;
        border-top: 1px solid #A0522D;
        font-size: 0.9em;
    }
    
    /* Styling Download Buttons - Aksen Sienna */
    .stDownloadButton button {
        background-color: #A0522D; /* Sienna */
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 15px;
        box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
        transition: background-color 0.3s;
    }
    .stDownloadButton button:hover {
        background-color: #8B4513; /* Saddle Brown saat hover */
    }
    
    /* Styling Metric Boxes (Statistik) */
    div[data-testid="stMetric"] {
        background-color: #FEF9E7; /* Krem Pucat */
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #A0522D;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stMetric"] label {
        color: #A0522D; /* Warna label metrik */
    }
    div[data-testid="stMetricValue"] {
        color: #654321; /* Warna nilai metrik (harus Dark Brown agar terbaca) */
    }

    /* Styling Expander */
    div[data-testid="stExpander"] {
        border: 1px solid #A0522D;
        border-radius: 8px;
    }
    div[data-testid="stExpander"] div[role="button"] {
        background-color: #FEF9E7; /* Latar belakang header expander */
        color: #4B4B4B !important;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True
)

# ============================================================
# Unduh dan muat model dari Hugging Face Hub
# ============================================================
REPO_ID = "yuliusat/JawaLens2.0"
MODEL_FILENAME = "Model.pkl"

with st.spinner("🔄 Memuat model ..."):
    try:
        # Mengunduh model dari repository Hugging Face
        MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
        
        # Memuat model menggunakan joblib
        model = joblib.load(MODEL_PATH)
        
        # Teks success dijamin terbaca karena CSS di atas
        st.success("Model berhasil dimuat! 🎉")
        
    except Exception as e:
        # Menampilkan pesan error jika pemuatan gagal
        st.error(f"Gagal memuat model: {e}")
        
        # Menghentikan eksekusi aplikasi Streamlit
        st.stop()
# ============================================================
# Konfigurasi Halaman
# ============================================================
st.set_page_config(page_title="JawaLens2.0", layout="wide")
# Judul utama menggunakan CSS di atas
st.markdown("<h1>Aplikasi Transliterasi Aksara Jawa</h1>", unsafe_allow_html=True) 

# ============================================================
# Lokasi folder hasil
# ============================================================
TEMP_FOLDER = tempfile.mkdtemp()
GDRIVE_FOLDER = os.path.join(TEMP_FOLDER, "JawaLens")
if not os.path.exists(GDRIVE_FOLDER):
    os.makedirs(GDRIVE_FOLDER, exist_ok=True)


# ============================================================
# Upload input
# ============================================================
uploaded_file = st.file_uploader("Unggah gambar naskah Aksara Jawa", type=["jpg", "png", "jpeg"])

if uploaded_file is None:
    st.warning("Silakan unggah gambar untuk memulai.")
    st.info("""
    ### 📋 Cara Penggunaan:
    1. Upload gambar naskah Aksara Jawa
    2. Tunggu proses otomatis (preprocessing, segmentasi, ekstraksi fitur, prediksi)
    3. Lihat hasil transliterasi
    4. Download hasil jika diperlukan
    """)
else:
    # simpan sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
        temp.write(uploaded_file.read())
        input_path = temp.name

    # siapkan folder output
    output_folder = os.path.join(GDRIVE_FOLDER, "Hasil_" + os.path.splitext(uploaded_file.name)[0])
    os.makedirs(output_folder, exist_ok=True)

    # tampilkan gambar
    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    with col1:
        img = Image.open(input_path)
        w, h = img.size
        # Perubahan untuk menampilkan gambar lebih besar namun tetap proporsional
        img_display = img.resize((max(1, w // 3), max(1, h // 3))) 
        st.image(img_display, caption="Gambar Asli", use_container_width=True)
    with col2:
        # Teks di info box sekarang kontras
        st.info(f"**File:** {uploaded_file.name}\n\n**Ukuran:** {w} x {h} px")

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    st.markdown("---")
    
    st.markdown("<h3>Proses Transliterasi Berjalan:</h3>", unsafe_allow_html=True)

    try:
        # ------------------------------------------------------------
        # Tahap 1: Segmentasi dan Preprocessing
        # ------------------------------------------------------------
        status_text.text("🔄 Tahap 1/5: Preprocessing dan Segmentasi...")
        progress_bar.progress(10)
        
        result_segment = backend.process_image(
            input_path=input_path,
            output_base_folder=output_folder,
            sigma_row=10,
            sigma_col=12
        )
        progress_bar.progress(20)
        # Teks success dijamin terbaca
        st.success(f"✅ Tahap 1 selesai: {len(result_segment)} karakter dari {result_segment['row_id'].nunique()} baris")

        # ------------------------------------------------------------
        # Tahap 2: Filtering
        # ------------------------------------------------------------
        status_text.text("🔄 Tahap 2/5: Filtering objek noise...")
        progress_bar.progress(30)
        
        df_results, df_saved = backend.process_and_save(
            result_segment,
            output_folder=os.path.join(output_folder, "Filtered"),
            method="manual",
            keep="larger",
            th=20,
            save_original=False
        )
        progress_bar.progress(40)
        # Teks success dijamin terbaca
        st.success(f"✅ Tahap 2 selesai: Dihapus {df_results['removed_objects'].sum()} objek noise")

        # ------------------------------------------------------------
        # Tahap 3: Cropping dan Normalisasi
        # ------------------------------------------------------------
        status_text.text("🔄 Tahap 3/5: Cropping dan Normalisasi...")
        progress_bar.progress(50)
        
        df_crop = backend.process_image_binary_1x1(
            df_results,
            binary_column="cleaned_binary_image",
            output_folder=os.path.join(output_folder, "Crop")
        )
        progress_bar.progress(55)
        
        df_rescale = backend.rescale_image_90x90(
            df_crop,
            name_column="Square_image_array",
            output_size=(90, 90),
            output_path=os.path.join(output_folder, "Rescale")
        )
        progress_bar.progress(60)
        # Teks success dijamin terbaca
        st.success("✅ Tahap 3 selesai: Normalisasi ke 90x90 pixels")

        # ------------------------------------------------------------
        # Tahap 4: Ekstraksi Fitur (8x8, proj_bins=16)
        # ------------------------------------------------------------
        status_text.text("🔄 Tahap 4/5: Ekstraksi Fitur...")
        progress_bar.progress(70)
        
        test_features_df = backend.batch_extract_to_dataframe(
            df_rescale["Processed_image_array_90X90"].tolist(),
            labels=None,
            out_size=(90, 90),
            zoning_grid=(8, 8),
            proj_bins=16
        )
        X_test = test_features_df.values
        progress_bar.progress(80)
        # Teks success dijamin terbaca
        st.success(f"✅ Tahap 4 selesai: {X_test.shape[1]} fitur per karakter")

        # ------------------------------------------------------------
        # Tahap 5: Prediksi Transliterasi
        # ------------------------------------------------------------
        status_text.text("🔄 Tahap 5/5: Prediksi dan Transliterasi...")
        progress_bar.progress(90)
        
        result_predict = backend.predict_image(X_test, MODEL_PATH)
        translit_text = backend.combine_latin_transliteration(result_predict)
        
        progress_bar.progress(100)
        status_text.text("Proses selesai! 🎉")

        # simpan hasil
        csv_path = os.path.join(output_folder, "hasil_fitur.csv")
        test_features_df.to_csv(csv_path, index=False)
        
        st.markdown("---")
        # tampilkan hasil transliterasi
        st.markdown("<h4>Hasil Transliterasi:</h4>", unsafe_allow_html=True)
        # Teks di text area sekarang kontras
        st.text_area("Teks Latin", translit_text, height=200)
        
        # Statistik
        st.markdown("---")
        st.markdown("<h4>Statistik Hasil:</h4>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            # Nilai metrik sekarang kontras
            st.metric("Total Baris", df_rescale['row_id'].nunique())
        with col2:
            st.metric("Total Karakter", len(df_rescale))
        with col3:
            st.metric("Karakter/Baris", f"{len(df_rescale) / df_rescale['row_id'].nunique():.1f}")
        with col4:
            st.metric("Total Kata", len(translit_text.split()))
        
        # Download buttons
        st.markdown("---")
        st.markdown("<h4>Download Hasil:</h4>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📥 Download Transliterasi (TXT)", 
                data=translit_text.encode('utf-8'), 
                file_name="hasil_transliterasi.txt",
                mime="text/plain"
            )
        with col2:
            st.download_button(
                "📥 Download Fitur (CSV)", 
                data=test_features_df.to_csv(index=False).encode('utf-8'), 
                file_name="hasil_fitur.csv",
                mime="text/csv"
            )
        with col3:
            detail_df = df_rescale[['row_id', 'col_id', 'start_row', 'end_row', 'start_col', 'end_col']].copy()
            detail_df['prediction'] = result_predict
            st.download_button(
                "📥 Download Detail (CSV)", 
                data=detail_df.to_csv(index=False).encode('utf-8'), 
                file_name="detail_prediksi.csv",
                mime="text/csv"
            )

        # Detail per baris
        with st.expander("Lihat Detail Prediksi per Baris"):
            for row_id in sorted(df_rescale['row_id'].unique()):
                row_data = df_rescale[df_rescale['row_id'] == row_id]
                predictions_in_row = [result_predict[i] for i in row_data.index]
                # Teks di expander sekarang kontras
                st.markdown(f"**Baris {row_id}:** {' '.join(predictions_in_row)}")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        import traceback
        st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
<div class='footer-style'>
    <p><b>JawaLens 2.0</b></p>
    <p>Menggunakan KNN dengan ekstraksi fitur Zoning (8x8), Projection Profile (16 bins), dan Hu Moments</p>
</div>
""", unsafe_allow_html=True)
