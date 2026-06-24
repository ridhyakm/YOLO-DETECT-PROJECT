import streamlit as st
from ultralytics import YOLO
import cv2

st.title("YOLO Dashboard (Upload Image/Video)")

model = YOLO("yolov8m.pt")

uploaded_file = st.file_uploader("Upload an image or video", type=["jpg","jpeg","png","mp4"])
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    if uploaded_file.type.startswith("image"):
        import numpy as np
        import PIL.Image
        image = PIL.Image.open(uploaded_file)
        results = model.predict(image, conf=0.5)
        st.image(results[0].plot(), caption="Detections")
    else:
        st.video(uploaded_file)
        st.write("Video uploaded — detection not live in cloud.")
