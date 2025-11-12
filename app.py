import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# ---------------------------------------------------------
# LOAD YOUR EXISTING DENSE MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mnist_model.h5")
    return model

model = load_model()

# ---------------------------------------------------------
# APP CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="MNIST Digit Classifier", page_icon="🧠", layout="centered")
st.title("🧠 MNIST Handwritten Digit Classifier")

st.markdown("""
Upload or draw a handwritten digit to get real-time predictions.  
This version uses your fully connected (Dense) model trained on MNIST.
""")

mode = st.sidebar.radio("Choose Input Mode:", ["Upload Image", "Draw Digit"])

# ---------------------------------------------------------
# PREDICTION FUNCTION (FOR DENSE MODEL)
# ---------------------------------------------------------
def preprocess_and_predict(img):
    # Convert to grayscale and resize
    image = ImageOps.grayscale(img)
    image = image.resize((28, 28))

    # Convert to numpy and normalize
    img_array = np.array(image).astype("float32") / 255.0

    # Invert if background is white (canvas case)
    if np.mean(img_array) > 0.5:
        img_array = 1 - img_array

    # 🔹 Apply thresholding to remove anti-alias noise
    img_array = np.where(img_array > 0.2, 1.0, 0.0)

    # 🔹 Center the digit like MNIST
    # Calculate bounding box of the digit
    rows = np.any(img_array, axis=1)
    cols = np.any(img_array, axis=0)
    if np.any(rows) and np.any(cols):
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        digit = img_array[rmin:rmax+1, cmin:cmax+1]
        # Resize cropped digit to fit 20x20 box, then pad to 28x28
        from cv2 import resize, copyMakeBorder, INTER_AREA, BORDER_CONSTANT
        digit_resized = resize(digit, (20, 20), interpolation=INTER_AREA)
        pad_h = (28 - 20) // 2
        pad_v = (28 - 20) // 2
        img_array = copyMakeBorder(digit_resized, pad_h, pad_h, pad_v, pad_v,
                                   BORDER_CONSTANT, value=0)

    # Detect blank image
    if np.mean(img_array) < 0.02:
        return None, None

    # Flatten for Dense model input (1, 784)
    img_array = img_array.reshape(1, 784)

    preds = model.predict(img_array)
    pred_digit = np.argmax(preds)
    confidence = np.max(preds)

    if confidence < 0.5:
        return "Uncertain", preds

    return pred_digit, preds


# ---------------------------------------------------------
# UPLOAD IMAGE MODE
# ---------------------------------------------------------
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=150)

        result, preds = preprocess_and_predict(image)

        if result is None:
            st.warning("No digit detected. Please upload a clearer image.")
        elif result == "Uncertain":
            st.warning("Model not confident enough. Try a clearer digit image.")
        else:
            st.success(f"### Predicted Digit: {result}")

            fig, ax = plt.subplots()
            ax.bar(range(10), preds[0])
            ax.set_xticks(range(10))
            ax.set_xlabel("Digit")
            ax.set_ylabel("Confidence")
            st.pyplot(fig)

# ---------------------------------------------------------
# DRAW DIGIT MODE
# ---------------------------------------------------------
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
        result, preds = preprocess_and_predict(image)

        if result is None:
            st.warning("No digit detected. Please draw something.")
        elif result == "Uncertain":
            st.warning("Model not confident enough. Draw the digit more clearly.")
        else:
            st.success(f"### Predicted Digit: {result}")

            fig, ax = plt.subplots()
            ax.bar(range(10), preds[0])
            ax.set_xticks(range(10))
            ax.set_xlabel("Digit")
            ax.set_ylabel("Confidence")
            st.pyplot(fig)
