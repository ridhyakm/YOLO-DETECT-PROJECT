import streamlit as st
from ultralytics import YOLO
import cv2
import pyttsx3

st.title("YOLO Real-Time Dashboard with Voice Alerts")

# Load YOLO model
model = YOLO("yolov8m.pt")

# Initialize voice engine
engine = pyttsx3.init()

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

    # Voice alerts for ALL unique detections
    spoken_labels = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            if conf > 0.7 and label not in spoken_labels:
                spoken_labels.append(label)

    # Speak detections outside Streamlit rendering
    if spoken_labels:
        for label in spoken_labels:
            engine.say(f"{label} detected")
        engine.runAndWait()
