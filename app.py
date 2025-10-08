# app.py - Aplikasi Streamlit Transliterasi Aksara Jawa
import streamlit as st
import os
import tempfile
import joblib
from PIL import Image
import backend  # Import semua fungsi dari backend.py
from huggingface_hub import hf_hub_download

# ============================================================
# Unduh dan muat model dari Hugging Face Hub
# ============================================================
REPO_ID = "yuliusat/JawaLens2.0"
MODEL_FILENAME = "Model.pkl"

with st.spinner("Memuat model ..."):
    try:
        # Mengunduh model dari repository Hugging Face
        MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
        
        # Memuat model menggunakan joblib
        model = joblib.load(MODEL_PATH)
        
        st.success("Model berhasil dimuat !")
        
    except Exception as e:
        # Menampilkan pesan error jika pemuatan gagal
        st.error(f"Gagal memuat model: {e}")
        
        # Menghentikan eksekusi aplikasi Streamlit
        st.stop()
# ============================================================
# Konfigurasi Halaman
# ============================================================
st.set_page_config(page_title="JawaLens2.0", layout="wide")
st.markdown("<h1 style='color:#850510;text-align:center;'>Aplikasi Transliterasi Aksara Jawa</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #850510;'>", unsafe_allow_html=True)

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
    col1, col2 = st.columns([1, 3])
    with col1:
        img = Image.open(input_path)
        w, h = img.size
        img_small = img.resize((max(1, w // 4), max(1, h // 4)))
        st.image(img_small, caption="Gambar Asli", use_container_width=True)
    with col2:
        st.info(f"**File:** {uploaded_file.name}\n\n**Ukuran:** {w} x {h} px")

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

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
        st.success(f"Tahap 1 selesai: {len(result_segment)} karakter dari {result_segment['row_id'].nunique()} baris")

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
        st.success(f"Tahap 2 selesai: Dihapus {df_results['removed_objects'].sum()} objek noise")

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
        st.success("Tahap 3 selesai: Normalisasi ke 90x90 pixels")

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
        st.success(f"Tahap 4 selesai: {X_test.shape[1]} fitur per karakter")

        # ------------------------------------------------------------
        # Tahap 5: Prediksi Transliterasi
        # ------------------------------------------------------------
        status_text.text("🔄 Tahap 5/5: Prediksi dan Transliterasi...")
        progress_bar.progress(90)
        
        result_predict = backend.predict_image(X_test, MODEL_PATH)
        translit_text = backend.combine_latin_transliteration(result_predict)
        
        progress_bar.progress(100)
        status_text.text("Proses selesai!")

        # simpan hasil
        csv_path = os.path.join(output_folder, "hasil_fitur.csv")
        test_features_df.to_csv(csv_path, index=False)

        # tampilkan hasil transliterasi
        st.markdown("<h4 style='color:#850510;'>Hasil Transliterasi:</h4>", unsafe_allow_html=True)
        st.text_area("Teks Latin", translit_text, height=200)
        
        # Statistik
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Baris", df_rescale['row_id'].nunique())
        with col2:
            st.metric("Total Karakter", len(df_rescale))
        with col3:
            st.metric("Karakter/Baris", f"{len(df_rescale) / df_rescale['row_id'].nunique():.1f}")
        with col4:
            st.metric("Total Kata", len(translit_text.split()))
        
        # Download buttons
        st.markdown("<h4 style='color:#850510;'>Download Hasil:</h4>", unsafe_allow_html=True)
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

        #st.success(f"Hasil lengkap tersimpan di: {output_folder}")
        
        # Detail per baris
        with st.expander("Lihat Detail Prediksi per Baris"):
            for row_id in sorted(df_rescale['row_id'].unique()):
                row_data = df_rescale[df_rescale['row_id'] == row_id]
                predictions_in_row = [result_predict[i] for i in row_data.index]
                st.markdown(f"**Baris {row_id}:** {' '.join(predictions_in_row)}")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        import traceback
        st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><b>JawaLens 2.0</b></p>
    <p>Menggunakan KNN dengan ekstraksi fitur Zoning (8x8), Projection Profile (16 bins), dan Hu Moments</p>
</div>
""", unsafe_allow_html=True)
