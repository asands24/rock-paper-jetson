import os, cv2
from flask import Flask, Response
from roboflow import Roboflow

# --- Config ---
PROJECT_NAME = "rock-paper-scissors-sxsw"
VERSION = 14
CONFIDENCE = 40   # %
OVERLAP = 30      # %
INPUT_SIZE = 416  # model input size
VIDEO_SOURCE = 0  # or "demo.mp4" for a prerecorded clip

# --- Roboflow init ---
api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise RuntimeError("Set ROBOFLOW_API_KEY (e.g., export ROBOFLOW_API_KEY=xxxx)")

print("loading Roboflow workspace...")
rf = Roboflow(api_key=api_key)
ws = rf.workspace()  # or rf.workspace("roboflow-58fyf") if you want to be explicit
print("loading Roboflow project...")
project = ws.project(PROJECT_NAME)
model = project.version(VERSION).model

# --- Video ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"Could not open VIDEO_SOURCE={VIDEO_SOURCE}")

def draw_prediction(frame, pred):
    x, y = pred["x"], pred["y"]
    w, h = pred["width"], pred["height"]
    cls = pred["class"]
    conf = pred["confidence"]

    x1 = int(x - w / 2); y1 = int(y - h / 2)
    x2 = int(x + w / 2); y2 = int(y + h / 2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{cls} ({conf:.2f})"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (0, 255, 0), -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def mjpeg_generator():
    print("Starting inference stream…")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Resize for the model
        resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        result = model.predict(resized, confidence=CONFIDENCE, overlap=OVERLAP).json()
        for pred in result.get("predictions", []):
            draw_prediction(resized, pred)

        # Encode as JPEG for MJPEG streaming
        ok, jpg = cv2.imencode(".jpg", resized)
        if not ok:
            continue
        frame_bytes = jpg.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

app = Flask(__name__)

@app.route("/")
def index():
    return """
    <html>
      <head><title>Rock • Paper • Scissors (Jetson)</title></head>
      <body style="background:#111;color:#eee;font-family:sans-serif;">
        <h2>Rock • Paper • Scissors (Roboflow v14)</h2>
        <p>Press Ctrl+C in the terminal to stop.</p>
        <img src="/video" style="max-width:98vw;">
      </body>
    </html>
    """

@app.route("/video")
def video():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    try:
        # Bind to all interfaces so you can open it from your laptop
        app.run(host="0.0.0.0", port=1143, threaded=True)
    finally:
        cap.release()

