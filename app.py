import streamlit as st
from ultralytics import YOLO
import cv2

st.title("YOLO Real-Time Dashboard")

# Load YOLO model
model = YOLO("yolov8m.pt")

# Open webcam
cap = cv2.VideoCapture(0)
stframe = st.empty()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO on each frame
    results = model.predict(frame, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    # Show live detections in browser
    stframe.image(annotated_frame, channels="BGR")
