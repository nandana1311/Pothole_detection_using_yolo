import streamlit as st
from ultralytics import YOLO
import os
from PIL import Image
import numpy as np
import io

st.title("Pothole Detection App")

st.write("This application uses a YOLO model to detect potholes in uploaded images.")
st.write("To use the app, simply upload an image using the file uploader below.")

# Define the HOME directory (assuming the script is run from the same location as the notebook)
HOME = os.getcwd()

# Load the trained model
model = YOLO('best.pt')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.success("Image uploaded successfully! Processing...")
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # To convert to PIL Image:
    image = Image.open(io.BytesIO(bytes_data))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform inference
    results = model.predict(image)

    # Display and save the results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        st.image(im, caption="Image with Predictions", use_column_width=True)
