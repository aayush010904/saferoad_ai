import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load your trained YOLOv8 model
model = YOLO("best.pt")

st.title("🚦 Saferoad AI (prototype)")
st.text("Note: Saferoad AI is still in development phase, we will be adding more functionalities soon. Stay Tuned!")

# File Uploader
uploaded_file = st.file_uploader("Upload an Image to detect if there is an accident or not", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to numpy array for YOLOv8
    image_np = np.array(image)

    # Perform Inference
    results = model(image_np)
    result_img = results[0].plot()

    # Show Results
    st.image(result_img, caption="Detected Objects", use_column_width=True)
