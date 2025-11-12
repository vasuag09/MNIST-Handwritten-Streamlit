import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mnist_model.h5")
    return model

model = load_model()

# App Title
st.set_page_config(page_title="MNIST Digit Classifier", page_icon="🧠", layout="centered")
st.title("🧠 MNIST Handwritten Digit Classifier")
st.markdown("""
Upload an image or draw a digit below to get real-time predictions.  
The model automatically converts, resizes, and normalizes your input.
""")

# Sidebar options
mode = st.sidebar.radio("Choose Input Mode:", ["Upload Image", "Draw Digit"])

# Prediction Function
def preprocess_and_predict(img):
    # Convert to grayscale
    image = ImageOps.grayscale(img)
    # Resize to 28x28
    image = image.resize((28, 28))
    # Convert to numpy array
    img_array = np.array(image).astype("float32") / 255.0

    # Auto invert if dark text on light background
    if np.mean(img_array) > 0.5:
        img_array = 1 - img_array

    img_array = img_array.reshape(1, 784)

    preds = model.predict(img_array)
    pred_digit = np.argmax(preds)
    return pred_digit, preds

# --- Upload Image Mode ---
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=150)
        pred_digit, preds = preprocess_and_predict(image)
        st.success(f"### Predicted Digit: {pred_digit}")

        # Confidence Plot
        fig, ax = plt.subplots()
        ax.bar(range(10), preds[0])
        ax.set_xticks(range(10))
        ax.set_xlabel("Digit")
        ax.set_ylabel("Confidence")
        st.pyplot(fig)

# --- Drawing Mode ---
elif mode == "Draw Digit":
    st.write("Draw a digit (0–9) below:")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    if canvas_result.image_data is not None:
        image = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype("uint8"))
        pred_digit, preds = preprocess_and_predict(image)
        st.success(f"### Predicted Digit: {pred_digit}")

        fig, ax = plt.subplots()
        ax.bar(range(10), preds[0])
        ax.set_xticks(range(10))
        ax.set_xlabel("Digit")
        ax.set_ylabel("Confidence")
        st.pyplot(fig)
