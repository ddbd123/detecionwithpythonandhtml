import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO  # Import YOLOv8

app = Flask(__name__)

# Load the YOLOv8 model (you can download pre-trained models or use your custom model)
model = YOLO('yolov8n.pt')  # Replace with your YOLOv8 model path if needed

detected_objects = []  # Globalna zmienna do przechowywania wykrytych obiektów

# Funkcja do detekcji obiektów za pomocą YOLOv8
def detect_objects(frame):
    global detected_objects
    detected_objects = []  # Reset listy wykrytych obiektów na każdą klatkę

    # YOLOv8 detekcja
    results = model(frame)  # Wykrywanie obiektów za pomocą YOLOv8

    # Pobierz wyniki i rysuj ramki na wykrytych obiektach
    for box in results[0].boxes:  # Iteracja po wykrytych obiektach
        # Pobierz współrzędne i etykiety
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Współrzędne ramki
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = f"{model.names[class_id]}: {confidence * 100:.2f}%"
        
        # Dodaj informacje o wykrytym obiekcie do listy
        detected_objects.append({
            'label': label,
            'coordinates': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        })

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

# Endpoint do zwracania danych o wykrytych obiektach
@app.route('/detected_objects')
def get_detected_objects():
    return jsonify(detected_objects)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
