import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = load_model('fruit_vegetable_transferlearning.h5')

class_names = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 
    'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 
    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

st.title('Fruit & Vegetable Classifier')
file = st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])

if file:
    img = Image.open(file).convert('RGB')
    st.image(img, use_container_width=True)
    
    x = preprocess_input(np.expand_dims(np.array(img.resize((224, 224))), axis=0))
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    
    st.success(f"**Prediction:** {class_names[idx].title()}")
    st.info(f"**Confidence:** {preds[idx] * 100:.2f}%")