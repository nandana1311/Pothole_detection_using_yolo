import streamlit as st
from ultralytics import YOLO
import os
from PIL import Image
import numpy as np
import io

st.title("Pothole Detection App")
st.write("This application uses a YOLO model to detect potholes in uploaded images.")
st.write("Upload an image using the file uploader below.")

# Load the trained model
model = YOLO('best.pt')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.success("Image uploaded successfully! Processing...")
    image = Image.open(io.BytesIO(uploaded_file.getvalue()))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL to numpy
    img_array = np.array(image)

    # Perform inference
    results = model.predict(img_array, conf=0.5, verbose=False)

    # Display predictions
    for r in results:
        im_array = r.plot()  # BGR array
        im = Image.fromarray(im_array[..., ::-1])  # Convert to RGB
        st.image(im, caption="Image with Predictions", use_column_width=True)
