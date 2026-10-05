import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# 1. Load the Saved Model
model = load_model('rice_image.keras') 

# 2. Define Rice Classes
class_names = [
    'Arborio', 
    'Basmati', 
    'Ipsala', 
    'Jasmine', 
    'Karacadag'
]

# Page Configuration & Header
st.title('🌾 Rice Variety Classification System')
st.write("Upload a rice grain image to detect its variety.")

# File Uploader
file = st.file_uploader('Upload Rice Grain Image', type=['jpg', 'jpeg', 'png'])

if file:
    # Read and Display the Image
    img = Image.open(file).convert('RGB')
    st.image(img, caption="Uploaded Rice Grain", use_container_width=True)
    
    img_resized = img.resize((64, 64)) 
    
    # Preprocessing
    x = np.array(img_resized, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)  # Shape: (1, 170, 170, 3)
    
    # Model Prediction
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    
    # Display Results
    st.success(f"**Predicted Variety:** {class_names[idx]}")
    st.info(f"**Confidence Rate:** {preds[idx] * 100:.2f}%")