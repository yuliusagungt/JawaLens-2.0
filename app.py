"""
JawaLens 2.0
======================================================

REQUIRED PACKAGES:
pip install streamlit streamlit-cropper pillow numpy opencv-python-headless joblib huggingface_hub scikit-image scipy scikit-learn

HOW TO RUN:
streamlit run app.py

DESCRIPTION:
A beautiful, minimal image processing app with interactive cropping.
Features a brick-red (#B7410E) and soft beige (#F5F0E1) color scheme.
Includes JawaLens 2.0 model for Javanese script transliteration.
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
import backend  # Import all functions from backend.py

# Try to import streamlit-cropper, provide fallback instructions
try:
    from streamlit_cropper import st_cropper
    CROPPER_AVAILABLE = True
except ImportError:
    CROPPER_AVAILABLE = False
    st.error("""
    ⚠️ **Missing Required Package**
    
    Please install streamlit-cropper:
    ```
    pip install streamlit-cropper
    ```
    Then restart the app.
    """)
    st.stop()

# ============================================================
# MODEL LOADING FROM HUGGING FACE
# ============================================================
REPO_ID = "yuliusat/JawaLens2"

# Model options
MODEL_OPTIONS = {
    "Model 1: 281 Kelas, 500 data per kelas, n3": "Model1.pkl",
    "Model 2: 281 Kelas, 500 data per kelas, n11": "Model2.pkl"
}

@st.cache_resource
def load_model(model_filename):
    """Load model from Hugging Face Hub with caching"""
    with st.spinner(f"Loading model {model_filename}..."):
        try:
            hf_token = st.secrets.get("HF_TOKEN", None)
            MODEL_PATH = hf_hub_download(
                repo_id=REPO_ID,
                filename=model_filename,
                token=hf_token
            )
            model = joblib.load(MODEL_PATH)
            st.success(f"Model {model_filename} loaded successfully!")
            return model, MODEL_PATH
        except Exception as e:
            st.error(f"Failed to load model: {e}")
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
# CUSTOM CSS - BRICK RED & BEIGE THEME
# ============================================================
st.markdown("""
<style>
    /* ── Warna utama ── */
    :root {
        --brick-red:   #8B1A00;   /* merah bata gelap */
        --brick-mid:   #A52800;   /* merah bata tengah */
        --brick-light: #C0390F;   /* merah bata terang */
        --beige:       #F5F0E1;   /* background beige */
        --beige-card:  #FDF8EF;   /* card sedikit lebih terang */
        --dark-brown:  #3E2723;
        --light-brown: #8D6E63;
        --text-main:   #2C1A10;
    }

    /* ── Background seluruh halaman ── */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: var(--beige) !important;
    }

    /* ── Sidebar (jika terbuka) ── */
    [data-testid="stSidebar"] {
        background-color: #EDE8D8 !important;
    }

    /* ── Header / toolbar atas ── */
    [data-testid="stHeader"] {
        background-color: var(--beige) !important;
    }

    /* ── Teks umum ── */
    .stApp, .stMarkdown, p, li, label {
        color: var(--text-main);
    }

    /* ── Heading ── */
    h1, h2, h3, h4 {
        color: var(--brick-red) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
    }

    /* ── Card container ── */
    .card {
        background: var(--beige-card);
        padding: 2rem;
        border-radius: 14px;
        border-left: 5px solid var(--brick-red);
        box-shadow: 0 3px 12px rgba(139, 26, 0, 0.12);
        margin: 1rem 0;
    }

    /* ── Tombol utama ── */
    .stButton > button {
        background-color: var(--brick-red) !important;
        color: #FFF5EE !important;
        border: none !important;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: background-color 0.25s ease, box-shadow 0.25s ease;
    }
    .stButton > button:hover {
        background-color: var(--brick-mid) !important;
        box-shadow: 0 4px 14px rgba(139, 26, 0, 0.35) !important;
    }
    .stButton > button:active {
        background-color: var(--brick-light) !important;
    }

    /* ── Tombol download ── */
    .stDownloadButton > button {
        background-color: var(--light-brown) !important;
        color: white !important;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        transition: background-color 0.25s ease;
    }
    .stDownloadButton > button:hover {
        background-color: var(--dark-brown) !important;
    }

    /* ── Selectbox & widget ── */
    [data-testid="stSelectbox"] > div > div {
        background-color: var(--beige-card) !important;
        border-color: var(--brick-red) !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background-color: var(--beige-card) !important;
        border: 2px dashed var(--brick-mid) !important;
        border-radius: 10px;
    }

    /* ── Tab aktif ── */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--brick-red) !important;
        border-bottom: 3px solid var(--brick-red) !important;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--light-brown);
    }

    /* ── Alert / info box ── */
    .stAlert {
        background-color: rgba(139, 26, 0, 0.06) !important;
        border-left: 4px solid var(--brick-red) !important;
        border-radius: 6px;
    }

    /* ── Progress bar ── */
    [data-testid="stProgressBar"] > div > div {
        background-color: var(--brick-red) !important;
    }

    /* ── Metric value ── */
    [data-testid="stMetricValue"] {
        color: var(--brick-red) !important;
        font-weight: 700;
    }

    /* ── Text area ── */
    .stTextArea textarea {
        background-color: var(--beige-card) !important;
        border-color: var(--brick-mid) !important;
        color: var(--text-main) !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background-color: var(--beige-card) !important;
        border: 1px solid rgba(139, 26, 0, 0.2) !important;
        border-radius: 8px;
    }

    /* ── Divider ── */
    hr {
        border-color: var(--brick-red);
        opacity: 0.25;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: var(--brick-red) !important;
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
# SESSION STATE INITIALIZATION
# ============================================================
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'cropped_image' not in st.session_state:
    st.session_state.cropped_image = None
if 'final_image' not in st.session_state:
    st.session_state.final_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'show_cropper' not in st.session_state:
    st.session_state.show_cropper = False
if 'javanese_results' not in st.session_state:
    st.session_state.javanese_results = None
if 'processing_mode' not in st.session_state:
    st.session_state.processing_mode = "simple"
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Model 1: 281 Kelas, 500 data per kelas, n3"
if 'model_path' not in st.session_state:
    st.session_state.model_path = None

# ============================================================
# MAIN APP
# ============================================================

# Header
st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>JawaLens 2.0</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8D6E63; margin-bottom: 2rem; font-size: 1.05rem;'>Transliterasi Aksara Jawa ke Latin berbasis Machine Learning</p>", unsafe_allow_html=True)
st.markdown("---")

# ============================================================
# MODEL SELECTION
# ============================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 🧠 Pilih Model")

selected_model = st.selectbox(
    "Pilih model yang akan digunakan:",
    options=list(MODEL_OPTIONS.keys()),
    index=list(MODEL_OPTIONS.keys()).index(st.session_state.selected_model),
    help="Pilih model KNN yang telah dilatih untuk transliterasi"
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
# STEP 1: UPLOAD OR CAPTURE IMAGE
# ============================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 📸 Step 1: Upload atau Ambil Foto")

tab1, tab2 = st.tabs(["📁 Upload File", "📷 Kamera"])

with tab1:
    uploaded_file = st.file_uploader(
        "Pilih file gambar (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        help="Upload gambar aksara Jawa dari perangkat kamu"
    )
    if uploaded_file is not None:
        st.session_state.uploaded_image = Image.open(uploaded_file)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Format", st.session_state.uploaded_image.format)
        with col2:
            st.metric("Ukuran", f"{st.session_state.uploaded_image.size[0]} x {st.session_state.uploaded_image.size[1]}")
        with col3:
            st.metric("Mode", st.session_state.uploaded_image.mode)
        st.success("Gambar berhasil diupload!")

with tab2:
    camera_image = st.camera_input(
        "Arahkan kamera ke dokumen aksara Jawa",
        help="Pastikan tulisan terlihat jelas dan pencahayaan cukup"
    )
    if camera_image is not None:
        st.session_state.uploaded_image = Image.open(camera_image)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Format", "JPEG")
        with col2:
            st.metric("Ukuran", f"{st.session_state.uploaded_image.size[0]} x {st.session_state.uploaded_image.size[1]}")
        with col3:
            st.metric("Mode", st.session_state.uploaded_image.mode)
        st.success("Foto berhasil diambil!")

if st.session_state.uploaded_image is None:
    if uploaded_file is None and camera_image is None:
        st.info("Upload gambar atau ambil foto untuk memulai.")
        st.session_state.cropped_image = None
        st.session_state.final_image = None
        st.session_state.processed_image = None

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# STEP 2: CROP OR PROCESS AS-IS
# ============================================================
if st.session_state.uploaded_image is not None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### ✂️ Step 2: Crop atau Langsung Proses")

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

    # ── Cropping Interface ──
    if st.session_state.show_cropper:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🖼️ Pilih Area Crop")
        st.info("Klik dan seret pada gambar untuk memilih area, lalu klik **Konfirmasi Crop**.")

        cropped_img = st_cropper(
            st.session_state.uploaded_image,
            realtime_update=True,
            box_color='#8B1A00',
            aspect_ratio=None,
            return_type='image'
        )

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("✅ Konfirmasi Crop", use_container_width=True, type="primary"):
                st.session_state.cropped_image = cropped_img
                st.session_state.final_image = cropped_img
                st.session_state.show_cropper = False
                st.success("Gambar berhasil di-crop!")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# STEP 3: JAVANESE SCRIPT PROCESSING
# ============================================================
if st.session_state.final_image is not None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🔍 Step 3: Transliterasi Aksara Jawa")
    st.info(f"Model aktif: **{st.session_state.selected_model}**")

    temp_folder = tempfile.mkdtemp()
    output_folder = os.path.join(temp_folder, "JawaLens_Results")

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Tahap 1/5: Preprocessing & Segmentasi...")
        progress_bar.progress(20)
        status_text.text("Tahap 2/5: Filter noise...")
        progress_bar.progress(40)
        status_text.text("Tahap 3/5: Crop & Normalisasi...")
        progress_bar.progress(60)
        status_text.text("Tahap 4/5: Ekstraksi Fitur...")
        progress_bar.progress(80)
        status_text.text("Tahap 5/5: Prediksi & Transliterasi...")

        results = process_javanese_script(
            st.session_state.final_image,
            st.session_state.model_path,
            output_folder
        )
        st.session_state.javanese_results = results

        progress_bar.progress(100)
        status_text.text("✅ Proses selesai!")
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Hasil Transliterasi ──
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📄 Hasil Transliterasi")
        st.image(st.session_state.final_image, caption="Gambar Aksara Jawa", use_container_width=True)

        st.markdown("#### Hasil Latin:")
        st.text_area("Transliterasi", results['transliteration'], height=200)

        st.markdown("#### Statistik:")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Baris", results['df_rescale']['row_id'].nunique())
        with col2:
            st.metric("Total Karakter", len(results['df_rescale']))
        with col3:
            avg_chars = len(results['df_rescale']) / results['df_rescale']['row_id'].nunique()
            st.metric("Karakter/Baris", f"{avg_chars:.1f}")
        with col4:
            st.metric("Total Kata", len(results['transliteration'].split()))
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Download ──
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 💾 Download Hasil")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.download_button(
                "📝 Transliterasi (TXT)",
                data=results['transliteration'].encode('utf-8'),
                file_name="hasil_transliterasi.txt",
                mime="text/plain",
                use_container_width=True
            )

        with col2:
            st.download_button(
                "📊 Fitur (CSV)",
                data=results['test_features_df'].to_csv(index=False).encode('utf-8'),
                file_name="hasil_fitur.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col3:
            detail_df = results['df_rescale'][['row_id', 'col_id', 'start_row', 'end_row', 'start_col', 'end_col']].copy()
            detail_df['prediction'] = results['result_predict']
            st.download_button(
                "🔎 Detail Prediksi (CSV)",
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
                label="🖼️ Semua Gambar (ZIP)",
                data=zip_buffer,
                file_name="hasil_gambar_jawalens.zip",
                mime="application/zip",
                use_container_width=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("🔍 Detail Prediksi per Baris"):
            for row_id in sorted(results['df_rescale']['row_id'].unique()):
                row_data = results['df_rescale'][results['df_rescale']['row_id'] == row_id]
                predictions_in_row = [results['result_predict'][i] for i in row_data.index]
                st.markdown(f"**Baris {row_id}:** {' '.join(predictions_in_row)}")

    except Exception as e:
        st.error(f"Error saat pemrosesan: {e}")
        import traceback
        st.code(traceback.format_exc())

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #8D6E63; padding: 2rem 0;'>
    <p><strong style="color:#8B1A00;">JawaLens 2.0</strong> &nbsp;|&nbsp; Built with Streamlit</p>
    <p style='font-size: 0.8rem;'>Transliterasi Aksara Jawa berbasis KNN · Zoning · Projection Profile · Hu Moments</p>
</div>
""", unsafe_allow_html=True)
