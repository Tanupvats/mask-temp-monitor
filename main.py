import cv2
import serial
import threading
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


SERIAL_PORT = 'COM5' 
BAUD_RATE = 115200

latest_temp = 0.0
system_running = True

def serial_listener():
    global latest_temp, system_running
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print(f"[*] Connected to microcontroller on {SERIAL_PORT}")
    except serial.SerialException:
        print(f"[!] Warning: Could not open {SERIAL_PORT}. Running vision-only mode.")
        ser = None

    while system_running and ser:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if line.startswith("TEMP:"):
                    latest_temp = float(line.split(":")[1])
        except Exception:
            pass


temp_thread = threading.Thread(target=serial_listener, daemon=True)
temp_thread.start()


print("[*] Loading Face Detector...")
face_net = cv2.dnn.readNet(
    "deploy.prototxt", 
    "res10_300x300_ssd_iter_140000.caffemodel"
)

print("[*] Loading Mask Classifier...")
mask_model = load_model("mask_detector.model")


def detect_and_predict_mask(frame, face_net, mask_model):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            
            face = frame[startY:endY, startX:endX]
            if face.size == 0: 
                continue
                
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))


    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = mask_model.predict(faces, batch_size=32)

    return (locs, preds)


print("[*] Starting Video Stream...")
cap = cv2.VideoCapture(0)
time.sleep(2.0) 

while True:
    ret, frame = cap.read()
    if not ret: 
        print("[!] Failed to grab frame from camera. Exiting...")
        break

    
    frame = cv2.resize(frame, (800, 600))

    
    (locs, preds) = detect_and_predict_mask(frame, face_net, mask_model)

    
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        
        temp_color = (0, 255, 0) if latest_temp < 37.5 else (0, 0, 255)
        temp_label = f"{latest_temp:.1f}C"
        if latest_temp >= 37.5:
            temp_label += " (FEVER)"

        
        display_text = f"{label}: {max(mask, withoutMask) * 100:.2f}% | Temp: {temp_label}"

        
        cv2.putText(frame, display_text, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    
    cv2.imshow("Secure Access Monitor", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


system_running = False
cap.release()
cv2.destroyAllWindows()