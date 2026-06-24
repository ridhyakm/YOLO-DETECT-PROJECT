from ultralytics import YOLO
import pyttsx3
import cv2

# 1. Load YOLO model (medium for better accuracy)
model = YOLO("yolov8m.pt")   # try yolov8l.pt for even better accuracy

# 2. Initialize voice engine
engine = pyttsx3.init()

# 3. Run detection on a sample image (optional showcase)
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    conf=0.5,   # only show detections above 50% confidence
    save=True
)

# 4. Live detection using webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO on each frame
    results = model.predict(frame, conf=0.5, verbose=False)

    # Show detections in a window
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Webcam Detection", annotated_frame)

    # Voice alerts for ALL unique detections in the frame
    spoken_labels = []  # track what we already said this frame
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])          # class index
            label = model.names[cls]       # class name
            conf = float(box.conf[0])      # confidence score
            if conf > 0.7 and label not in spoken_labels:
                engine.say(f"{label} detected with {conf*100:.1f} percent confidence")
                spoken_labels.append(label)

    # Speak all detections together
    if spoken_labels:
        engine.runAndWait()

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
