import os
import json
import joblib
import numpy as np
from PIL import Image, ImageEnhance
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_FILE  = "model_pipeline1.pkl"       
JSON_FILE   = "500_labels_for_testing.json"   
IMAGES_DIR  = "test_images"  

def preprocess_image(img_path):
    """Mirrors the exact pipeline used during training."""
    img = Image.open(img_path).convert('L') 
    img_array = np.array(img).flatten()  
    return img_array

def extract_hog_features(images_flat):
    """Same HOG config as training."""
    features = []
    for img_flat in images_flat:
        img_2d = img_flat.reshape(28, 28)
        feat = hog(img_2d, pixels_per_cell=(7, 7),
                   cells_per_block=(2, 2), feature_vector=True)
        features.append(feat)
    return np.array(features)

print("Loading model...")
svm = joblib.load(MODEL_FILE)
print(f"  Model loaded: {type(svm).__name__}")

with open(JSON_FILE) as f:
    labels = json.load(f)
print(f"  Labels loaded: {len(labels)} entries")

print(f"\nLoading images from '{IMAGES_DIR}'...")
raw_images, y, missing = [], [], []

for key, label in labels.items():
    img_path = os.path.join(IMAGES_DIR, f"{key}.bmp")

    if not os.path.exists(img_path):
        missing.append(key)
        continue

    raw_images.append(preprocess_image(img_path))
    y.append(int(label))

if missing:
    print(f"  Warning: {len(missing)} images not found and skipped: {missing[:5]}{'...' if len(missing) > 5 else ''}")

print(f"  Loaded {len(raw_images)} images")

print("Extracting HOG features...")
X = extract_hog_features(raw_images)
y = np.array(y)
print(f"  Feature matrix shape: {X.shape}")

print("\nRunning predictions...")
y_pred = svm.predict(X).astype(int)

acc = accuracy_score(y, y_pred)
n_correct = int(acc * len(y))

print(f"\n{'='*40}")
print(f"  Accuracy : {acc:.2%}  ({n_correct}/{len(y)})")
print(f"{'='*40}\n")
