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
REPO_ID = "yuliusat/JawaLens2.0"

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
            # Download model from Hugging Face repository
            MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename=model_filename)
            
            # Load model using joblib
            model = joblib.load(MODEL_PATH)
            
            st.success(f"Model {model_filename} loaded successfully!")
            return model, MODEL_PATH
            
        except Exception as e:
            # Display error message if loading fails
            st.error(f"Failed to load model: {e}")
            st.stop()
            return None, None

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="JawaLens2.0",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS - BRICK RED THEME
# ============================================================
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --brick-red: #B7410E;
        --soft-beige: #F5F0E1;
        --dark-brown: #3E2723;
        --light-brown: #8D6E63;
    }
    
    /* Global background */
    .main {
        background-color: var(--soft-beige);
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: var(--brick-red) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
    }
    
    /* Card containers */
    .card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(183, 65, 14, 0.1);
        margin: 1rem 0;
    }
    
    /* Custom buttons */
    .stButton > button {
        background-color: var(--brick-red);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #8B310A;
        box-shadow: 0 4px 12px rgba(183, 65, 14, 0.3);
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: var(--dark-brown);
        font-weight: 500;
    }
    
    /* File uploader */
    .stFileUploader > label {
        color: var(--dark-brown);
        font-weight: 500;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: rgba(183, 65, 14, 0.05);
        border-left: 4px solid var(--brick-red);
        border-radius: 4px;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background-color: var(--light-brown);
        color: white;
        border-radius: 8px;
        padding: 0.75rem 2rem;
    }
    
    .stDownloadButton > button:hover {
        background-color: var(--dark-brown);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: var(--brick-red);
    }
    
    /* Divider */
    hr {
        border-color: var(--brick-red);
        opacity: 0.3;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def process_image(image_pil, processing_type="grayscale"):
    """
    Apply simple image processing.
    Options: grayscale, edge_detection, blur, sharpen
    """
    img_array = np.array(image_pil)
    
    # Convert RGB to BGR for OpenCV
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
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        processed = cv2.filter2D(img_bgr, -1, kernel)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    else:
        processed = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(processed)


def process_javanese_script(image_pil, model_path, output_folder):
    """
    Process Javanese script image using JawaLens backend.
    Returns transliteration results and processed dataframes.
    """
    # Save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
        image_pil.save(temp, format="PNG")
        input_path = temp.name
    
    try:
        # Prepare output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Stage 1: Segmentation and Preprocessing
        result_segment = backend.process_image(
            input_path=input_path,
            output_base_folder=output_folder,
            sigma_row=10,
            sigma_col=12
        )
        
        # Stage 2: Filtering
        df_results, df_saved = backend.process_and_save(
            result_segment,
            output_folder=os.path.join(output_folder, "Filtered"),
            method="manual",
            keep="larger",
            th=20,
            save_original=False
        )
        
        # Stage 3: Cropping and Normalization
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
        
        # Stage 4: Feature Extraction
        test_features_df = backend.batch_extract_to_dataframe(
            df_rescale["Processed_image_array_90X90"].tolist(),
            labels=None,
            out_size=(90, 90),
            zoning_grid=(8, 8),
            proj_bins=16
        )
        X_test = test_features_df.values
        
        # Stage 5: Prediction and Transliteration
        result_predict = backend.predict_image(X_test, model_path)
        translit_text = backend.combine_latin_transliteration(result_predict)
        
        # Save results
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
        # Clean up temp file
        if os.path.exists(input_path):
            os.remove(input_path)


def pil_to_bytes(image_pil, format="PNG"):
    """Convert PIL Image to bytes for download."""
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
    st.session_state.processing_mode = "simple"  # simple or javanese
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Model 1: 281 Kelas, 500 data per kelas, n3"
if 'model_path' not in st.session_state:
    st.session_state.model_path = None

# ============================================================
# MAIN APP
# ============================================================

# Header
st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>JawaLens2.0</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8D6E63; margin-bottom: 2rem;'>Upload, crop, and process your images with style</p>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# MODEL SELECTION
# ============================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### Select Model")

selected_model = st.selectbox(
    "Choose the model to use:",
    options=list(MODEL_OPTIONS.keys()),
    index=list(MODEL_OPTIONS.keys()).index(st.session_state.selected_model),
    help="Select which trained model to use for transliteration"
)

# Load model if selection changed
if selected_model != st.session_state.selected_model:
    st.session_state.selected_model = selected_model
    model_filename = MODEL_OPTIONS[selected_model]
    model, MODEL_PATH = load_model(model_filename)
    st.session_state.model_path = MODEL_PATH
else:
    # Load model for first time or if not loaded yet
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
st.markdown("### Step 1: Upload or Capture Your Image")

# Tab untuk memilih metode input
tab1, tab2 = st.tabs(["📁 Upload File", "📷 Capture Photo"])

with tab1:
    uploaded_file = st.file_uploader(
        "Choose an image file (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        help="Select an image from your device"
    )

    if uploaded_file is not None:
        # Load and store image
        st.session_state.uploaded_image = Image.open(uploaded_file)
        
        # Display uploaded image info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Format", st.session_state.uploaded_image.format)
        with col2:
            st.metric("Size", f"{st.session_state.uploaded_image.size[0]} x {st.session_state.uploaded_image.size[1]}")
        with col3:
            st.metric("Mode", st.session_state.uploaded_image.mode)
        
        st.success("Image uploaded successfully!")

with tab2:
    st.markdown("""
    <style>
    .camera-container {
        position: relative;
        width: 100%;
        max-width: 640px;
        margin: 0 auto;
    }
    .guide-rectangle {
        position: absolute;
        top: 10%;
        left: 10%;
        width: 80%;
        height: 80%;
        border: 3px solid #B7410E;
        border-radius: 8px;
        box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.5);
        pointer-events: none;
        z-index: 10;
    }
    .guide-text {
        position: absolute;
        top: 5%;
        left: 50%;
        transform: translateX(-50%);
        color: white;
        background-color: rgba(183, 65, 14, 0.8);
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: 500;
        z-index: 11;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Camera input widget
    camera_image = st.camera_input(
        "Position your document within the guide rectangle and capture",
        help="Align your Javanese script document within the red rectangle guide"
    )
    
    if camera_image is not None:
        # Load and store image from camera
        st.session_state.uploaded_image = Image.open(camera_image)
        
        # Display captured image info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Format", "JPEG")
        with col2:
            st.metric("Size", f"{st.session_state.uploaded_image.size[0]} x {st.session_state.uploaded_image.size[1]}")
        with col3:
            st.metric("Mode", st.session_state.uploaded_image.mode)
        
        st.success("Photo captured successfully!")

# Handle case when no image is provided
if st.session_state.uploaded_image is None:
    if uploaded_file is None and camera_image is None:
        st.info("Please upload an image or capture a photo to get started")
        st.session_state.cropped_image = None
        st.session_state.final_image = None
        st.session_state.processed_image = None

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# STEP 2: CROP OR PROCESS AS-IS
# ============================================================
if st.session_state.uploaded_image is not None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Step 2: Crop or Process")
    
    # Choice: Crop or Process as-is
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Crop Image", use_container_width=True):
            st.session_state.show_cropper = True
            st.session_state.final_image = None
            st.rerun()
    
    with col2:
        if st.button("Direct Image Processing", use_container_width=True):
            st.session_state.show_cropper = False
            st.session_state.final_image = st.session_state.uploaded_image
            st.session_state.cropped_image = None
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ============================================================
    # CROPPING INTERFACE
    # ============================================================
    if st.session_state.show_cropper:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Interactive Cropping")
        
        st.info("""
        **How to crop:**
        - Click and drag on the image to select an area
        - Adjust the selection by dragging the corners or edges
        - Click 'Confirm Crop' when you're happy with the selection
        """)
        
        # Cropper widget
        cropped_img = st_cropper(
            st.session_state.uploaded_image,
            realtime_update=True,
            box_color='#B7410E',
            aspect_ratio=None,
            return_type='image'
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Confirm Crop", use_container_width=True, type="primary"):
                st.session_state.cropped_image = cropped_img
                st.session_state.final_image = cropped_img
                st.session_state.show_cropper = False
                st.success("Image cropped successfully!")
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# STEP 3: JAVANESE SCRIPT PROCESSING
# ============================================================
if st.session_state.final_image is not None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Step 3: Javanese Script Transliteration")
    
    # Display selected model info
    st.info(f"Using: **{st.session_state.selected_model}**")
    
    # Create temp folder for results
    temp_folder = tempfile.mkdtemp()
    output_folder = os.path.join(temp_folder, "JawaLens_Results")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Stage 1/5: Preprocessing and Segmentation...")
        progress_bar.progress(20)
        
        status_text.text("Stage 2/5: Filtering noise...")
        progress_bar.progress(40)
        
        status_text.text("Stage 3/5: Cropping and Normalization...")
        progress_bar.progress(60)
        
        status_text.text("Stage 4/5: Feature Extraction...")
        progress_bar.progress(80)
        
        status_text.text("Stage 5/5: Transliteration...")
        
        # Process with JawaLens using selected model
        results = process_javanese_script(
            st.session_state.final_image,
            st.session_state.model_path,
            output_folder
        )
        st.session_state.javanese_results = results
        
        progress_bar.progress(100)
        status_text.text("Processing complete!")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display Results
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Transliteration Results")
        
        # Display original image
        st.image(st.session_state.final_image, caption="Original Javanese Script", use_container_width=True)
        
        # Display transliteration
        st.markdown("#### Latin Transliteration:")
        st.text_area("Result", results['transliteration'], height=200)
        
        # Statistics
        st.markdown("#### Statistics:")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", results['df_rescale']['row_id'].nunique())
        with col2:
            st.metric("Total Characters", len(results['df_rescale']))
        with col3:
            avg_chars = len(results['df_rescale']) / results['df_rescale']['row_id'].nunique()
            st.metric("Chars/Row", f"{avg_chars:.1f}")
        with col4:
            st.metric("Total Words", len(results['transliteration'].split()))
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Download Section
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Download Hasil")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.download_button(
                "Download Transliterasi (TXT)",
                data=results['transliteration'].encode('utf-8'),
                file_name="hasil_transliterasi.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                "Download Fitur (CSV)",
                data=results['test_features_df'].to_csv(index=False).encode('utf-8'),
                file_name="hasil_fitur.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            detail_df = results['df_rescale'][['row_id', 'col_id', 'start_row', 'end_row', 'start_col', 'end_col']].copy()
            detail_df['prediction'] = results['result_predict']
            st.download_button(
                "Download Detail (CSV)",
                data=detail_df.to_csv(index=False).encode('utf-8'),
                file_name="detail_prediksi.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # ============================================================
        # Membuat ZIP dari seluruh hasil gambar
        # ============================================================
        with col4:
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                # Loop semua file gambar di subfolder output
                for root, _, files in os.walk(results['output_folder']):
                    for file in files:
                        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                            full_path = os.path.join(root, file)
                            # Menyimpan dalam struktur folder yang ringkas
                            arcname = os.path.relpath(full_path, results['output_folder'])
                            zipf.write(full_path, arcname)
            # Reset posisi pointer agar bisa dibaca ulang
            zip_buffer.seek(0)
            
            # ============================================================
            # Tombol download ZIP
            # ============================================================
            st.download_button(
                label="Download Semua Gambar (ZIP)",
                data=zip_buffer,
                file_name="hasil_gambar_jawalens.zip",
                mime="application/zip",
                use_container_width=True
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Detail per row
        with st.expander("View Detailed Predictions per Row"):
            for row_id in sorted(results['df_rescale']['row_id'].unique()):
                row_data = results['df_rescale'][results['df_rescale']['row_id'] == row_id]
                predictions_in_row = [results['result_predict'][i] for i in row_data.index]
                st.markdown(f"**Row {row_id}:** {' '.join(predictions_in_row)}")
    
    except Exception as e:
        st.error(f"Error during processing: {e}")
        import traceback
        st.code(traceback.format_exc())

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #8D6E63; padding: 2rem 0;'>
    <p><strong>JawaLens 2.0</strong> | Built with Streamlit</p>
    <p style='font-size: 0.8rem;'>Javanese Script Transliteration powered by KNN with Zoning, Projection Profile & Hu Moments</p>
</div>
""", unsafe_allow_html=True)
