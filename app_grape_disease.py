import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Load the Saved Model
model = load_model('grape_disease.keras')  

# 2. Define Grape Leaf Classes 
class_names = [
    'Black_rot', 
    'ESCA_(Black_Measles)', 
    'Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Healthy'
]

# Page Configuration & Header
st.title('🍇 Grape Disease Detection System')
st.write("Upload a grape leaf image to detect potential diseases.")

# File Uploader
file = st.file_uploader('Upload Grape Leaf Image', type=['jpg', 'jpeg', 'png'])

if file:
    # Read and Display the Image
    img = Image.open(file).convert('RGB')
    st.image(img, caption="Uploaded Grape Leaf", use_container_width=True)
    
    img_resized = img.resize((170, 170)) 
    
    # 4. Standard CNN Normalization
    x = np.array(img_resized, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)  # Shape: (1, 170, 170, 3)
    
    # 5. Model Prediction
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    
    # 6. Display Results
    st.success(f"**Predicted Status:** {class_names[idx]}")
    st.info(f"**Confidence Rate:** {preds[idx] * 100:.2f}%")