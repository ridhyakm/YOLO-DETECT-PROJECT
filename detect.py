from ultralytics import YOLO

# Load a pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")

# Run detection on a single image
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    save=True
)

# (Optional) Run detection on webcam
# results = model.predict(source=0, show=True)
