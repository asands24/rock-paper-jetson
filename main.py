import cv2
from roboflow import Roboflow
import supervision as sv
import os

# Load API Key
api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise Exception("Set your ROBOFLOW_API_KEY as an environment variable!")

rf = Roboflow(api_key=api_key)
project = rf.workspace().project("rock-paper-scissors-sxsw")
model = project.version(14).model

# Video source
cap = cv2.VideoCapture(0)  # or replace with your camera index

box_annotator = sv.BoxAnnotator()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize to match input size (416x416)
    resized = cv2.resize(frame, (416, 416))

    # Predict
    result = model.predict(resized, confidence=40, overlap=30).json()
    detections = sv.Detections.from_inference(result)
    labels = [f"{d['class']} ({d['confidence']:.2f})" for d in result["predictions"]]

    annotated = box_annotator.annotate(resized.copy(), detections=detections, labels=labels)
    cv2.imshow("Rock Paper Scissors", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

