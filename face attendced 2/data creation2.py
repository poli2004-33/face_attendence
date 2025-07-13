# part1_capture_faces.py

import cv2
import os
import time
import imutils
import csv

# Load Haar cascade
cascade_path = r'C:\Users\jagad\Downloads\haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(cascade_path)

if detector.empty():
    raise IOError(f"Cannot load cascade from {cascade_path}")

# Get user input
name = input("Enter your name: ")
roll_number = input("Enter your roll number: ")

# Create dataset directory
dataset_dir = 'datasets'
user_dir = os.path.join(dataset_dir, name)
os.makedirs(user_dir, exist_ok=True)

# Save student info
info = [name, roll_number]
with open('student.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(info)

# Start webcam capture
print("Starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)

total = 0
while total < 50:
    ret, frame = cam.read()
    if not ret:
        break

    frame_resized = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_img = frame_resized[y:y+h, x:x+w]
        save_path = os.path.join(user_dir, f"{str(total).zfill(5)}.png")
        cv2.imwrite(save_path, face_img)
        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
        total += 1

    cv2.imshow("Capturing Faces", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
print("Face image capture complete.")
