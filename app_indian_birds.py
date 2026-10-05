import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model('indian_bird.keras') 

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

st.title('Indian Bird Species Classifier')
file = st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])

if file:
    img = Image.open(file).convert('RGB')
    st.image(img, use_container_width=True)
    
    img_resized = img.resize((32, 32))
    
    x = np.array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)
    
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    
    st.success(f"**Prediction:** {class_names[idx]}")
    st.info(f"**Confidence:** {preds[idx] * 100:.2f}%")