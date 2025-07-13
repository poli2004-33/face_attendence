# part3_train_recognizer.py

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import os

# File paths
embeddingFile = "output/embeddings.pickle"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"

# Step 1: Load face embeddings
print("🔄 Loading face embeddings...")
if not os.path.exists(embeddingFile):
    print(f"❌ Embedding file not found at {embeddingFile}")
    exit()

try:
    with open(embeddingFile, "rb") as f:
        data = pickle.load(f)
    print("✅ Embeddings loaded successfully.")
except Exception as e:
    print(f"❌ Error loading embeddings: {e}")
    exit()

# Step 2: Validate data
if "embeddings" not in data or "names" not in data:
    print("❌ Embedding file structure is invalid.")
    exit()

print(f"📊 Total embeddings: {len(data['embeddings'])}")
print(f"👤 Unique names: {set(data['names'])}")

if len(set(data["names"])) <= 1:
    print("❌ At least two different identities are needed for training.")
    exit()

# Step 3: Encode labels
print("🔠 Encoding labels...")
try:
    labelEnc = LabelEncoder()
    labels = labelEnc.fit_transform(data["names"])
    print(f"✅ Labels encoded. Classes: {labelEnc.classes_}")
except Exception as e:
    print(f"❌ Error encoding labels: {e}")
    exit()

# Step 4: Train recognizer (SVM)
print("🎓 Training SVM model...")
try:
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)
    print("✅ Model trained successfully.")
except Exception as e:
    print(f"❌ Error training model: {e}")
    exit()

# Step 5: Save recognizer model
print("💾 Saving recognizer...")
try:
    with open(recognizerFile, "wb") as f:
        pickle.dump(recognizer, f)
    print(f"✅ Recognizer saved to {recognizerFile}.")
except Exception as e:
    print(f"❌ Error saving recognizer: {e}")

# Step 6: Save label encoder
print("💾 Saving label encoder...")
try:
    with open(labelEncFile, "wb") as f:
        pickle.dump(labelEnc, f)
    print(f"✅ Label encoder saved to {labelEncFile}.")
except Exception as e:
    print(f"❌ Error saving label encoder: {e}")
