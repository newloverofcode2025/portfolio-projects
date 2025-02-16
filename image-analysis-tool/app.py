import streamlit as st
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

# Import our modules (we'll create these next)
from src.detection import object_detector
from src.face_analysis import face_analyzer
from src.enhancement import image_enhancer
from src.utils import image_utils

def main():
    st.set_page_config(
        page_title="AI Image Analysis Tool",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 AI Image Analysis Tool")
    st.write("Upload an image to analyze it using state-of-the-art AI models!")

    # Sidebar for task selection
    st.sidebar.title("Analysis Tasks")
    task = st.sidebar.selectbox(
        "Choose a task",
        ["Object Detection", "Face Analysis", "Image Enhancement", "All"]
    )

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert uploaded file to image
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Display original image
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        # Process based on selected task
        if task in ["Object Detection", "All"]:
            st.subheader("Object Detection")
            # We'll implement these functions next
            detected_objects = object_detector.detect(image_np)
            st.image(detected_objects, use_column_width=True)

        if task in ["Face Analysis", "All"]:
            st.subheader("Face Analysis")
            face_results = face_analyzer.analyze(image_np)
            st.image(face_results, use_column_width=True)

        if task in ["Image Enhancement", "All"]:
            st.subheader("Enhanced Image")
            enhanced_image = image_enhancer.enhance(image_np)
            st.image(enhanced_image, use_column_width=True)

if __name__ == "__main__":
    main()
