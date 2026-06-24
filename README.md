# 🚀 YOLO Object Detection Project

This project uses **YOLOv8** for object detection.  
It has two parts:
- **Online dashboard (Streamlit Cloud)** → Upload images/videos and see detections in your browser.  
- **Local run (Python script)** → Real‑time webcam detection with voice alerts.

---

## 🌐 Online Dashboard
Try it here: https://yolo-detect-project-f7s7v3cmcy2jihuh8c5cvg.streamlit.app/

### Features
- Upload an image or video.  
- See bounding boxes and labels for detected objects.  
- Runs completely online, no installation needed.

---

## 💻 Local Run (Webcam + Voice Alerts)
To use the full version with webcam and voice alerts:

1. Clone the repo:
   ```bash
   git clone https://github.com/ridhyakm/YOLO-DETECT-PROJECT.git
   cd YOLO-DETECT-PROJECT
Install requirements:

bash
pip install -r requirements.txt
Run detection:

bash
python detect.py
Features
Real‑time webcam detection.

Voice alerts announce detected objects.

Works only on your laptop (not in the cloud).

📦 Requirements
The project uses:

ultralytics (YOLOv8)

opencv-python-headless (for cloud)

opencv-python (for local webcam)

streamlit (dashboard)

pyttsx3 (voice alerts, local only)

🎯 Summary
Online dashboard → Upload images/videos, shareable link.

Local run → Webcam + voice alerts.
