import streamlit as st
import numpy as np
import time
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM CSS (STYLISH DESIGN)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Indian Birds AI | Species Classifier",
    page_icon="🦜",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MODEL LOADING (CACHED FOR PERFORMANCE)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_bird_model():
    return load_model('indian_bird_transferlearning.keras')

with st.spinner('🚀 Loading Transfer Learning Model...'):
    model = load_bird_model()

# -----------------------------------------------------------------------------
# 3. CLASS NAMES (25 BIRD SPECIES)
# -----------------------------------------------------------------------------
class_names = [
    'Asian Green Bee-Eater', 'Brown-Headed Barbet', 'Cattle Egret', 
    'Common Kingfisher', 'Common Myna', 'Common Rosefinch', 
    'Common Tailorbird', 'Coppersmith Barbet', 'Forest Wagtail', 
    'Gray Wagtail', 'Hoopoe', 'House Crow', 'Indian Grey Hornbill', 
    'Indian Peacock', 'Indian Pitta', 'Indian Roller', 'Jungle Babbler', 
    'Northern Lapwing', 'Red-Wattled Lapwing', 'Ruddy Shelduck', 
    'Rufous Treepie', 'Sarus Crane', 'White Wagtail', 
    'White-Breasted Kingfisher', 'White-Breasted Waterhen'
]

# -----------------------------------------------------------------------------
# 4. SIDEBAR METRICS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📊 Model Metrics")
    st.markdown("---")
    st.metric(label="Architecture", value="MobileNetV2")
    st.metric(label="Accuracy Rate", value="98.29%", delta="+94.40% vs CNN")
    st.metric(label="Input Resolution", value="224 x 224 px")
    st.markdown("---")
    st.caption("Deep Learning & Transfer Learning Project")

# -----------------------------------------------------------------------------
# 5. MAIN INTERFACE
# -----------------------------------------------------------------------------
st.markdown('<p class="main-title">🦜 Indian Bird Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Powered by <b>MobileNetV2 Transfer Learning</b> • 98.29% Accuracy</p>', unsafe_allow_html=True)

# Image Upload Area
file = st.file_uploader("Upload or drag & drop a bird image:", type=['jpg', 'jpeg', 'png'])

if file:
    img = Image.open(file).convert('RGB')
    
    # Display Uploaded Image
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Preprocessing (224x224 & MobileNetV2 Format)
    img_resized = img.resize((224, 224))
    x = np.array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # Required MobileNetV2 preprocessing
    
    # Prediction & Animation
    with st.spinner('🦚 Analyzing bird species...'):
        time.sleep(0.3)  # Brief delay for UX effect
        preds = model.predict(x)[0]
        idx = np.argmax(preds)
        confidence = preds[idx] * 100

    top_pred = class_names[idx]

    # Results Display
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"### 🦅 Predicted Species\n**{top_pred}**")
        
    with col2:
        st.info(f"### 🎯 Confidence Score\n**{confidence:.2f}%**")

    # Top-3 Predictions Section
    st.markdown("### 🔝 Top 3 Predictions")
    top_3_indices = np.argsort(preds)[-3:][::-1]
    
    for i in top_3_indices:
        species_name = class_names[i]
        score = preds[i] * 100
        st.write(f"**{species_name}** ({score:.2f}%)")
        st.progress(int(score))