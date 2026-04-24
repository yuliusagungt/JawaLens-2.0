"""
JawaLens 2.0
======================================================

REQUIRED PACKAGES:
pip install streamlit streamlit-cropper pillow numpy opencv-python-headless joblib huggingface_hub scikit-image scipy scikit-learn

HOW TO RUN:
streamlit run app.py
"""

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import joblib
import tempfile
import os
import zipfile
from huggingface_hub import hf_hub_download
import backend

try:
    from streamlit_cropper import st_cropper
    CROPPER_AVAILABLE = True
except ImportError:
    CROPPER_AVAILABLE = False
    st.error("Paket streamlit-cropper tidak ditemukan. Jalankan: pip install streamlit-cropper")
    st.stop()

# ============================================================
# MODEL LOADING FROM HUGGING FACE
# ============================================================
REPO_ID = "yuliusat/JawaLens2"

MODEL_OPTIONS = {
    "Model 1 — 281 Kelas, 500 data/kelas, n=3": "Model1.pkl",
    "Model 2 — 281 Kelas, 500 data/kelas, n=11": "Model2.pkl"
}

@st.cache_resource
def load_model(model_filename):
    with st.spinner(f"Memuat model {model_filename}..."):
        try:
            hf_token = st.secrets.get("HF_TOKEN", None)
            MODEL_PATH = hf_hub_download(
                repo_id=REPO_ID,
                filename=model_filename,
                token=hf_token
            )
            model = joblib.load(MODEL_PATH)
            st.success(f"Model {model_filename} berhasil dimuat.")
            return model, MODEL_PATH
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            st.stop()
            return None, None

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="JawaLens 2.0",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS — ELEGANT MINIMALIST
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=DM+Sans:wght@300;400;500&display=swap');

    :root {
        --red:        #7A1500;
        --red-mid:    #9C1F00;
        --red-hover:  #5E1000;
        --beige:      #F4EFE3;
        --beige-card: #FAF7F1;
        --beige-dark: #EDE6D6;
        --brown-text: #2A1A10;
        --muted:      #8A7060;
        --border:     rgba(122, 21, 0, 0.15);
    }

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: var(--beige) !important;
        font-family: 'DM Sans', sans-serif;
        color: var(--brown-text);
    }

    [data-testid="stHeader"],
    [data-testid="stToolbar"] {
        background-color: var(--beige) !important;
    }

    [data-testid="stSidebar"] {
        background-color: var(--beige-dark) !important;
    }

    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 4rem !important;
        max-width: 900px !important;
    }

    /* Typography */
    h1 {
        font-family: 'Cormorant Garamond', Georgia, serif !important;
        font-size: 2.8rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px;
        color: var(--red) !important;
        line-height: 1.15;
    }
    h2, h3 {
        font-family: 'Cormorant Garamond', Georgia, serif !important;
        font-weight: 600 !important;
        color: var(--red) !important;
    }
    h4 {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.75rem !important;
        letter-spacing: 1.8px !important;
        text-transform: uppercase !important;
        color: var(--muted) !important;
    }
    p, li, label, .stMarkdown {
        font-family: 'DM Sans', sans-serif;
        color: var(--brown-text);
        font-weight: 300;
        line-height: 1.7;
    }

    /* Section card */
    .section {
        background: var(--beige-card);
        border: 1px solid var(--border);
        border-radius: 3px;
        padding: 2rem 2.2rem;
        margin: 1.4rem 0;
    }
    .section-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.68rem;
        font-weight: 500;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.25rem;
    }
    .section-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.55rem;
        font-weight: 600;
        color: var(--red);
        margin-bottom: 1.4rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid var(--border);
    }

    hr {
        border: none;
        border-top: 1px solid var(--border) !important;
        margin: 1.5rem 0;
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--red) !important;
        color: #FFF8F5 !important;
        border: none !important;
        border-radius: 2px !important;
        padding: 0.65rem 1.8rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        letter-spacing: 1.2px !important;
        text-transform: uppercase !important;
        transition: background-color 0.2s ease !important;
        box-shadow: none !important;
    }
    .stButton > button:hover {
        background-color: var(--red-hover) !important;
        box-shadow: none !important;
    }

    /* Download buttons */
    .stDownloadButton > button {
        background-color: transparent !important;
        color: var(--red) !important;
        border: 1px solid var(--border) !important;
        border-radius: 2px !important;
        padding: 0.6rem 1.1rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.74rem !important;
        font-weight: 500 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        transition: background-color 0.2s ease, border-color 0.2s ease !important;
    }
    .stDownloadButton > button:hover {
        background-color: rgba(122, 21, 0, 0.05) !important;
        border-color: var(--red) !important;
    }

    /* Selectbox */
    [data-testid="stSelectbox"] > div > div {
        background-color: var(--beige-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 2px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.88rem !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: var(--beige-card) !important;
        border: 1px dashed rgba(122, 21, 0, 0.25) !important;
        border-radius: 3px !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid var(--border) !important;
        gap: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.76rem !important;
        font-weight: 400 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        color: var(--muted) !important;
        padding: 0.6rem 1.4rem !important;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--red) !important;
        font-weight: 500 !important;
        border-bottom: 2px solid var(--red) !important;
    }

    /* Alert */
    .stAlert {
        background-color: rgba(122, 21, 0, 0.04) !important;
        border: 1px solid var(--border) !important;
        border-left: 3px solid var(--red) !important;
        border-radius: 2px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.86rem !important;
    }

    /* Progress */
    [data-testid="stProgressBar"] > div > div {
        background-color: var(--red) !important;
    }
    [data-testid="stProgressBar"] > div {
        background-color: var(--beige-dark) !important;
    }

    /* Metric */
    [data-testid="stMetricValue"] {
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 2.1rem !important;
        color: var(--red) !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.68rem !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        color: var(--muted) !important;
    }

    /* Text area */
    .stTextArea textarea {
        background-color: var(--beige-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 2px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
        color: var(--brown-text) !important;
        line-height: 1.8;
    }
    .stTextArea textarea:focus {
        border-color: var(--red) !important;
        box-shadow: none !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background-color: var(--beige-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 2px !important;
    }
    [data-testid="stExpander"] summary {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.76rem !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        color: var(--muted) !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def process_image(image_pil, processing_type="grayscale"):
    img_array = np.array(image_pil)
    if len(img_array.shape) == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array

    if processing_type == "grayscale":
        processed = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    elif processing_type == "edge_detection":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    elif processing_type == "blur":
        processed = cv2.GaussianBlur(img_bgr, (15, 15), 0)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    elif processing_type == "sharpen":
        kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        processed = cv2.filter2D(img_bgr, -1, kernel)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    else:
        processed = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(processed)


def process_javanese_script(image_pil, model_path, output_folder):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
        image_pil.save(temp, format="PNG")
        input_path = temp.name

    try:
        os.makedirs(output_folder, exist_ok=True)

        result_segment = backend.process_image(
            input_path=input_path,
            output_base_folder=output_folder,
            sigma_row=10,
            sigma_col=12
        )

        df_results, df_saved = backend.process_and_save(
            result_segment,
            output_folder=os.path.join(output_folder, "Filtered"),
            method="manual",
            keep="larger",
            th=20,
            save_original=False
        )

        df_crop = backend.process_image_binary_1x1(
            df_results,
            binary_column="cleaned_binary_image",
            output_folder=os.path.join(output_folder, "Crop")
        )

        df_rescale = backend.rescale_image_90x90(
            df_crop,
            name_column="Square_image_array",
            output_size=(90, 90),
            output_path=os.path.join(output_folder, "Rescale")
        )

        test_features_df = backend.batch_extract_to_dataframe(
            df_rescale["Processed_image_array_90X90"].tolist(),
            labels=None,
            out_size=(90, 90),
            zoning_grid=(8, 8),
            proj_bins=16
        )
        X_test = test_features_df.values

        result_predict = backend.predict_image(X_test, model_path)
        translit_text = backend.combine_latin_transliteration(result_predict)

        csv_path = os.path.join(output_folder, "hasil_fitur.csv")
        test_features_df.to_csv(csv_path, index=False)

        return {
            'transliteration': translit_text,
            'df_rescale': df_rescale,
            'test_features_df': test_features_df,
            'result_predict': result_predict,
            'output_folder': output_folder
        }

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


def pil_to_bytes(image_pil, format="PNG"):
    buf = io.BytesIO()
    image_pil.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    'uploaded_image': None,
    'cropped_image': None,
    'final_image': None,
    'processed_image': None,
    'show_cropper': False,
    'javanese_results': None,
    'processing_mode': 'simple',
    'selected_model': list(MODEL_OPTIONS.keys())[0],
    'model_path': None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1.2rem;">
    <h1>JawaLens 2.0</h1>
    <p style="color:#8A7060; font-size:0.9rem; font-weight:300; letter-spacing:1px;
              text-transform:uppercase; margin-top:0.5rem;">
        Transliterasi Aksara Jawa ke Latin berbasis Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# MODEL SELECTION
# ============================================================
st.markdown("""
<div class="section">
    <div class="section-label">Konfigurasi</div>
    <div class="section-title">Pilih Model</div>
""", unsafe_allow_html=True)

selected_model = st.selectbox(
    "Model yang digunakan",
    options=list(MODEL_OPTIONS.keys()),
    index=list(MODEL_OPTIONS.keys()).index(st.session_state.selected_model),
    help="Pilih model KNN yang telah dilatih untuk transliterasi aksara Jawa"
)

if selected_model != st.session_state.selected_model:
    st.session_state.selected_model = selected_model
    model_filename = MODEL_OPTIONS[selected_model]
    model, MODEL_PATH = load_model(model_filename)
    st.session_state.model_path = MODEL_PATH
else:
    if st.session_state.model_path is None:
        model_filename = MODEL_OPTIONS[selected_model]
        model, MODEL_PATH = load_model(model_filename)
        st.session_state.model_path = MODEL_PATH
    else:
        MODEL_PATH = st.session_state.model_path

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# STEP 1 — INPUT GAMBAR
# ============================================================
st.markdown("""
<div class="section">
    <div class="section-label">Langkah 1</div>
    <div class="section-title">Masukkan Gambar</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Upload File", "Kamera"])

with tab1:
    uploaded_file = st.file_uploader(
        "Pilih file gambar (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        help="Upload gambar aksara Jawa dari perangkat"
    )
    if uploaded_file is not None:
        st.session_state.uploaded_image = Image.open(uploaded_file)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Format", st.session_state.uploaded_image.format or "—")
        with col2:
            st.metric("Ukuran", f"{st.session_state.uploaded_image.size[0]} × {st.session_state.uploaded_image.size[1]}")
        with col3:
            st.metric("Mode", st.session_state.uploaded_image.mode)
        st.success("Gambar berhasil dimuat.")

with tab2:
    camera_image = st.camera_input(
        "Arahkan kamera ke dokumen aksara Jawa",
        help="Pastikan tulisan terlihat jelas dengan pencahayaan yang cukup"
    )
    if camera_image is not None:
        st.session_state.uploaded_image = Image.open(camera_image)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Format", "JPEG")
        with col2:
            st.metric("Ukuran", f"{st.session_state.uploaded_image.size[0]} × {st.session_state.uploaded_image.size[1]}")
        with col3:
            st.metric("Mode", st.session_state.uploaded_image.mode)
        st.success("Foto berhasil diambil.")

if st.session_state.uploaded_image is None:
    if uploaded_file is None and camera_image is None:
        st.info("Upload gambar atau ambil foto untuk memulai proses transliterasi.")
        st.session_state.cropped_image = None
        st.session_state.final_image = None
        st.session_state.processed_image = None

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# STEP 2 — CROP
# ============================================================
if st.session_state.uploaded_image is not None:
    st.markdown("""
    <div class="section">
        <div class="section-label">Langkah 2</div>
        <div class="section-title">Crop atau Langsung Proses</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Crop Gambar", use_container_width=True):
            st.session_state.show_cropper = True
            st.session_state.final_image = None
            st.rerun()
    with col2:
        if st.button("Langsung Proses", use_container_width=True):
            st.session_state.show_cropper = False
            st.session_state.final_image = st.session_state.uploaded_image
            st.session_state.cropped_image = None
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.show_cropper:
        st.markdown("""
        <div class="section">
            <div class="section-label">Area Seleksi</div>
            <div class="section-title">Pilih Area Crop</div>
        """, unsafe_allow_html=True)

        st.info("Seret sudut atau tepi seleksi untuk menyesuaikan area, kemudian konfirmasi.")

        cropped_img = st_cropper(
            st.session_state.uploaded_image,
            realtime_update=True,
            box_color='#7A1500',
            aspect_ratio=None,
            return_type='image'
        )

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Konfirmasi Crop", use_container_width=True, type="primary"):
                st.session_state.cropped_image = cropped_img
                st.session_state.final_image = cropped_img
                st.session_state.show_cropper = False
                st.success("Area crop dikonfirmasi.")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# STEP 3 — TRANSLITERASI
# ============================================================
if st.session_state.final_image is not None:
    st.markdown("""
    <div class="section">
        <div class="section-label">Langkah 3</div>
        <div class="section-title">Transliterasi Aksara Jawa</div>
    """, unsafe_allow_html=True)

    st.info(f"Model aktif: {st.session_state.selected_model}")

    temp_folder = tempfile.mkdtemp()
    output_folder = os.path.join(temp_folder, "JawaLens_Results")

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Tahap 1 / 5 — Preprocessing dan segmentasi...")
        progress_bar.progress(20)
        status_text.text("Tahap 2 / 5 — Filter noise...")
        progress_bar.progress(40)
        status_text.text("Tahap 3 / 5 — Crop dan normalisasi...")
        progress_bar.progress(60)
        status_text.text("Tahap 4 / 5 — Ekstraksi fitur...")
        progress_bar.progress(80)
        status_text.text("Tahap 5 / 5 — Prediksi dan transliterasi...")

        results = process_javanese_script(
            st.session_state.final_image,
            st.session_state.model_path,
            output_folder
        )
        st.session_state.javanese_results = results

        progress_bar.progress(100)
        status_text.text("Selesai.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Hasil
        st.markdown("""
        <div class="section">
            <div class="section-label">Keluaran</div>
            <div class="section-title">Hasil Transliterasi</div>
        """, unsafe_allow_html=True)

        st.image(st.session_state.final_image, caption="Gambar Aksara Jawa", use_container_width=True)

        st.markdown("<h4>Teks Latin</h4>", unsafe_allow_html=True)
        st.text_area("Transliterasi", results['transliteration'], height=200, label_visibility="collapsed")

        st.markdown("<h4>Statistik</h4>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Baris", results['df_rescale']['row_id'].nunique())
        with col2:
            st.metric("Total Karakter", len(results['df_rescale']))
        with col3:
            avg_chars = len(results['df_rescale']) / results['df_rescale']['row_id'].nunique()
            st.metric("Karakter / Baris", f"{avg_chars:.1f}")
        with col4:
            st.metric("Total Kata", len(results['transliteration'].split()))

        st.markdown("</div>", unsafe_allow_html=True)

        # Download
        st.markdown("""
        <div class="section">
            <div class="section-label">Ekspor</div>
            <div class="section-title">Unduh Hasil</div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.download_button(
                "Transliterasi (TXT)",
                data=results['transliteration'].encode('utf-8'),
                file_name="hasil_transliterasi.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col2:
            st.download_button(
                "Fitur (CSV)",
                data=results['test_features_df'].to_csv(index=False).encode('utf-8'),
                file_name="hasil_fitur.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col3:
            detail_df = results['df_rescale'][['row_id', 'col_id', 'start_row', 'end_row', 'start_col', 'end_col']].copy()
            detail_df['prediction'] = results['result_predict']
            st.download_button(
                "Detail Prediksi (CSV)",
                data=detail_df.to_csv(index=False).encode('utf-8'),
                file_name="detail_prediksi.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col4:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(results['output_folder']):
                    for file in files:
                        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                            full_path = os.path.join(root, file)
                            arcname = os.path.relpath(full_path, results['output_folder'])
                            zipf.write(full_path, arcname)
            zip_buffer.seek(0)
            st.download_button(
                "Semua Gambar (ZIP)",
                data=zip_buffer,
                file_name="hasil_gambar_jawalens.zip",
                mime="application/zip",
                use_container_width=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Detail Prediksi per Baris"):
            for row_id in sorted(results['df_rescale']['row_id'].unique()):
                row_data = results['df_rescale'][results['df_rescale']['row_id'] == row_id]
                predictions_in_row = [results['result_predict'][i] for i in row_data.index]
                st.markdown(f"**Baris {row_id}** &nbsp; {' · '.join(predictions_in_row)}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat pemrosesan: {e}")
        import traceback
        st.code(traceback.format_exc())


# ============================================================
# FOOTER
# ============================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding: 1.5rem 0 2.5rem; color: #8A7060;">
    <p style="font-family:'Cormorant Garamond',serif; font-size:1.15rem;
              color:#7A1500; margin-bottom:0.4rem; font-weight:600;">
        JawaLens 2.0
    </p>
    <p style="font-size:0.7rem; letter-spacing:1.5px; text-transform:uppercase; margin:0;">
        KNN &nbsp;&middot;&nbsp; Zoning &nbsp;&middot;&nbsp; Projection Profile &nbsp;&middot;&nbsp; Hu Moments
    </p>
</div>
""", unsafe_allow_html=True)
