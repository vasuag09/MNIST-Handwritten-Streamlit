# 🧠 MNIST Handwritten Digit Classifier – Streamlit Deployment

An interactive web app that predicts handwritten digits using a trained **Dense Neural Network (DNN)** model on the **MNIST dataset**.  
Built with **TensorFlow** and deployed using **Streamlit**, this project allows users to either **upload an image** or **draw digits directly** on the canvas for real-time predictions.

---

## 🚀 Live Demo
🎥 [Watch Demo Video](#)  
🌐 [Try it on Streamlit Cloud](https://your-app-link.streamlit.app)

---

## 🧩 Features
- 🖼️ **Upload or Draw Digits** – Supports image upload or canvas-based drawing.  
- 🧠 **Custom Dense Model** – Uses a fully connected neural network trained on MNIST.  
- 🧹 **Automatic Preprocessing** – Inverts, normalizes, thresholds, and centers digits to match MNIST format.  
- ⚠️ **Smart Detection** – Warns when the image is blank or model confidence is low.  
- 💻 **Deployed on Streamlit Cloud** – Fully accessible via web browser.

---

## 🧠 Model Overview
| Layer | Type | Activation | Output Shape |
|--------|------|-------------|---------------|
| Dense | 128 neurons | ReLU | (128,) |
| Dense | 128 neurons | ReLU | (128,) |
| Dense | 10 neurons | Softmax | (10,) |

- **Optimizer:** SGD  
- **Loss:** Categorical Crossentropy  
- **Dataset:** MNIST (28×28 grayscale digits)  
- **Accuracy:** ~97% on test data

---

## 🧮 Preprocessing Pipeline
1. Convert uploaded image to grayscale.  
2. Resize to 28×28 pixels.  
3. Invert colors if necessary (white background).  
4. Apply binary thresholding.  
5. Crop and center the digit to match MNIST alignment.  
6. Normalize pixel values to [0,1].  
7. Flatten to (1, 784) for DNN input.

---

## 📦 Tech Stack
- **Frontend/UI:** Streamlit  
- **Backend Model:** TensorFlow / Keras  
- **Image Processing:** Pillow, OpenCV  
- **Visualization:** Matplotlib  
- **Hosting:** Streamlit Cloud  

---

## 🧰 Installation

```bash
git clone https://github.com/vasuag09/mnist-handwritten-streamlit.git
cd mnist-handwritten-streamlit
pip install -r requirements.txt
streamlit run app.py
```

---

### 📄 Requirements
streamlit
tensorflow
pillow
numpy
matplotlib
opencv-python-headless
streamlit-drawable-canvas

### 📊 Project Highlights

Full deployment pipeline from model training to interactive inference.
Advanced preprocessing ensuring consistent predictions.
Modular and stable app design (macOS + Streamlit Cloud optimized).

### 🧑‍💻 Author

Vasu Agrawal
AI / ML Developer | Data Science Student | Web Engineer
🔗 LinkedIn

💻 GitHub