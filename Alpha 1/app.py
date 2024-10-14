import os
import cv2
import numpy as np
from flask import Flask, render_template, Response
from ultralytics import YOLO  # Import YOLOv8

app = Flask(__name__)

# Load the YOLOv8 model (you can download pre-trained models or use your custom model)
model = YOLO('yolov8n.pt')  # Replace with your YOLOv8 model path if needed

# Funkcja do detekcji obiektów za pomocą YOLOv8
def detect_objects(frame):
    # YOLOv8 detekcja
    results = model(frame)  # Wykrywanie obiektów za pomocą YOLOv8

    # Pobierz wyniki i rysuj ramki na wykrytych obiektach
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Pobierz współrzędne i etykiety
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Współrzędne ramki
            label = f"{box.cls[0]}: {box.conf[0] * 100:.2f}%"
            
            # Rysowanie prostokątów i etykiet na klatce
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Funkcja do strumieniowania wideo z detekcją obiektów
def generate_video():
    cap = cv2.VideoCapture(0)  # Użycie kamerki
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Wykrywanie obiektów na każdej klatce
        frame = detect_objects(frame)

        # Kodowanie obrazu jako JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()

        # Zwracanie klatki jako strumień
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

    cap.release()

# Strona główna
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint do strumieniowania wideo
@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
