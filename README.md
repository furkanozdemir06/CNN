# 🧠 CNN Image Classification Projects

**Can a computer tell a Basmati grain from a Jasmine one? Name an Indian bird from a single photo? Spot a diseased grape leaf before a farmer does?**

This repository is my hands-on journey into computer vision. I trained neural networks to recognize everything from **fruits and vegetables** to **bird species**, **rice varieties** and **grape leaf diseases**, and turned the best models into small apps you can try yourself.

Every project asks the same question: *what works better, a CNN built from scratch or a pre-trained model?* The answers were surprising.

## 🔍 What's Inside

### 🍎 Fruit & Vegetable Classification
Can a model tell apart **36 different fruits and vegetables**? My custom CNN barely got started, but transfer learning with MobileNetV2 hit **96.97%** accuracy with a clean confusion matrix across all classes.
👉 [Explore the notebook](Fruit_Veg_CNN.ipynb)

### 🍇 Grape Disease Detection
Early disease detection can save a harvest. A custom CNN already reached 91%, and MobileNetV2 with higher-resolution inputs pushed it to **99%**.
👉 [Explore the notebook](Grape_Disease_CNN.ipynb)

### 🦜 Indian Birds Classification
Birds are hard: the differences hide in plumage patterns and beak shapes. Going from tiny 32x32 images to 224x224 with a pre-trained backbone took accuracy from **3.89% to 98.29%**.
👉 [Explore the notebook](Indian_Birds_CNN.ipynb)

### 🍚 Rice Variety Classification
Five rice varieties (Arborio, Basmati, Ipsala, Jasmine and Karacadag) that look almost identical to the human eye. MobileNetV2 reached **99.49%**, just ahead of my custom CNN at 98.56%.
👉 [Explore the notebook](Rice_CNN.ipynb)


## 🏆 The Big Picture

| Project | Custom CNN | MobileNetV2 |
| :--- | :---: | :---: |
| Fruits & Vegetables (36 classes) | 3.53% | **96.97%** |
| Grape Disease | 91% | **99%** |
| Indian Birds | 3.89% | **98.29%** |
| Rice Varieties (5 classes) | 98.56% | **99.49%** |

## 💡 What I Learned

- **Transfer learning is a superpower.** Pre-trained ImageNet features gave huge gains, especially on hard, fine-grained datasets.
- **Resolution matters.** Tiny images erase the details that separate similar classes.
- **Simple problems need simple models.** On the rice dataset, even a custom CNN came close to the best result.



## 🛠️ Built With

Python · TensorFlow/Keras · MobileNetV2 · NumPy · Matplotlib · Streamlit

