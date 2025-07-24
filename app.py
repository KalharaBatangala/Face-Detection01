import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Title of the Streamlit app
st.title("Face Detection App")
st.write("Upload an image to detect faces and count the number of people.")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Load OpenCV's pre-trained Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Convert back to PIL Image for display
    result_image = Image.fromarray(image_np)

    # Display the image with detected faces
    st.image(result_image, caption="Image with Detected Faces", use_column_width=True)

    # Display the count of detected faces
    st.write(f"Number of people detected: **{len(faces)}**")