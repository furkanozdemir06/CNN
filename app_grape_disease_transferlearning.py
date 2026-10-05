import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Grape Vision AI | Leaf Disease Detector",
    page_icon="🍇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS STYLING ---
st.markdown("""
    <style>
    /* Main Background & Font Styling */
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Header Container */
    .header-box {
        padding: 1.5rem;
        background: linear-gradient(135deg, #1e1b4b 0%, #311b92 100%);
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .header-title {
        color: #ffffff;
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-sub {
        color: #a5b4fc;
        font-size: 1.05rem;
        margin-top: 0.4rem;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #334155;
        backdrop-filter: blur(10px);
    }
    
    /* Custom Progress Bar Color */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #6366f1 , #a855f7);
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. MODEL LOADING WITH CACHING ---
@st.cache_resource
def load_grape_model():
    return load_model('grape_disease_transferlearning.keras')

with st.spinner("🚀 Loading AI Model..."):
    model = load_grape_model()

# Class labels
class_names = [
    'Black Rot', 
    'ESCA (Black Measles)', 
    'Leaf Blight (Isariopsis Spot)', 
    'Healthy'
]

# --- 4. SIDEBAR PANEL ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/grapes.png", width=70)
    st.title("GrapeVision AI")
    st.caption("Deep Learning Diagnostic Engine")
    st.markdown("---")
    
    st.markdown("### 📊 Model Info")
    st.markdown("""
    * **Architecture:** MobileNetV2
    * **Accuracy:** `99.0%`
    * **Target:** Grape Leaf Pathology
    """)
    
    st.markdown("---")
    st.info("💡 **Tip:** Upload high-resolution leaf images under direct lighting for optimal diagnostic precision.")

# --- 5. HEADER SECTION ---
st.markdown("""
    <div class="header-box">
        <h1 class="header-title">🍇 Grape Leaf Pathology Detector</h1>
        <p class="header-sub">Instant AI-Powered Plant Disease Identification & Health Analytics</p>
    </div>
""", unsafe_allow_html=True)

# --- 6. FILE UPLOADER & PROCESSING ---
file = st.file_uploader('Drag & Drop or Browse Grape Leaf Image', type=['jpg', 'jpeg', 'png'])

if file is not None:
    # Read Image
    img = Image.open(file).convert('RGB')
    
    # 2-Column Dashboard Layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📸 Input Image")
        st.image(img, use_container_width=True, caption="Uploaded Leaf Specimen")
        
    with col2:
        st.markdown("### 🤖 AI Diagnostic Result")
        
        # Preprocessing (MobileNetV2 Transfer Learning Setup)
        img_resized = img.resize((224, 224))
        x = np.array(img_resized, dtype=np.float32)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Prediction
        with st.spinner('Analyzing cellular patterns...'):
            preds = model.predict(x)[0]
            idx = np.argmax(preds)
            confidence = preds[idx] * 100
            predicted_class = class_names[idx]
            
        # Dynamic Result Banner
        if predicted_class == 'Healthy':
            st.success(f"### 🌱 Diagnosis: **{predicted_class}**")
        else:
            st.error(f"### ⚠️ Diagnosis: **{predicted_class}**")
            
        st.metric(label="Prediction Confidence", value=f"{confidence:.2f}%")
        
        st.markdown("---")
        st.markdown("#### 📈 Class Probabilities Distribution")
        
        # Probability Bars
        for i, class_name in enumerate(class_names):
            prob = preds[i] * 100
            st.write(f"**{class_name}** ({prob:.1f}%)")
            st.progress(int(prob))
            
else:
    # Default Empty State Banner
    st.info("👆 Please upload a grape leaf image to run diagnosis.")