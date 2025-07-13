# part4_realtime_recognition.py

import pickle
import numpy as np
import imutils
import cv2
import os
import time

# Set paths to the models and trained files
datasets_path = r"C:\pythonfiles\face attendence"
recognizer_file = os.path.join(datasets_path, "recognizer.pickle")
label_enc_file = os.path.join(datasets_path, "label_encoder.pickle")
face_detector_prototxt = os.path.join(datasets_path, "deploy.prototxt")
face_detector_model = os.path.join(datasets_path, "res10_300x300_ssd_iter_140000.caffemodel")
embedding_model = os.path.join(datasets_path, "openface_nn4.small2.v1.t7")

# Load the recognizer and label encoder
def load_pickle(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        exit()

recognizer = load_pickle(recognizer_file)
le = load_pickle(label_enc_file)

# Load models
try:
    detector = cv2.dnn.readNetFromCaffe(face_detector_prototxt, face_detector_model)
    embedder = cv2.dnn.readNetFromTorch(embedding_model)
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# Confidence threshold
conf_threshold = 0.5

# Start webcam
print("🎥 Starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # Prepare input blob for face detector
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0),
                                 swapRB=False, crop=False)
    detector.setInput(blob)
    detections = detector.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            (fh, fw) = face.shape[:2]

            if fh < 20 or fw < 20:
                continue

            # Create face embedding
            face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                              (96, 96), (0, 0, 0),
                                              swapRB=True, crop=True)
            embedder.setInput(face_blob)
            vec = embedder.forward()

            # Predict
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # Draw results
            text = f"{name}: {proba * 100:.2f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cam.release()
cv2.destroyAllWindows()
