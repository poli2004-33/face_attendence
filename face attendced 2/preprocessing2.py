# part2_generate_embeddings.py

from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os
from deepface import DeepFace

# Config
dataset = "datasets"
output_dir = "output"
embedding_file = os.path.join(output_dir, "embeddings.pickle")
prototxt = r"C:\pythonfiles\face attendence\deploy.prototxt"
model = r"C:\pythonfiles\face attendence\res10_300x300_ssd_iter_140000.caffemodel"
conf_threshold = 0.5

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load DNN face detector
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

# Prepare variables
imagePaths = list(paths.list_images(dataset))
knownEmbeddings = []
knownNames = []
total = 0

# Loop through all images
for (i, imagePath) in enumerate(imagePaths):
    print(f"Processing image {i + 1}/{len(imagePaths)}: {imagePath}")
    name = os.path.basename(os.path.dirname(imagePath))
    image = cv2.imread(imagePath)
    if image is None:
        print(f"Skipping corrupted image: {imagePath}")
        continue

    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0), swapRB=False)

    detector.setInput(blob)
    detections = detector.forward()

    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]

            if face.shape[0] < 20 or face.shape[1] < 20:
                continue

            try:
                face = cv2.resize(face, (224, 224))  # Required size for VGG-Face
                embedding = DeepFace.represent(face, model_name='VGG-Face')[0]['embedding']
                knownEmbeddings.append(embedding)
                knownNames.append(name)
                total += 1
            except Exception as e:
                print(f"Error processing {imagePath}: {e}")

print(f"Total valid embeddings: {total}")

# Save embeddings to file
data = {"embeddings": knownEmbeddings, "names": knownNames}
with open(embedding_file, "wb") as f:
    pickle.dump(data, f)
print(f"Embeddings saved to {embedding_file}")
