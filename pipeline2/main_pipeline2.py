import os
import random
import numpy as np
from PIL import Image, ImageEnhance
import json
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from check_accuracy import check_accuracy
import joblib

SEED_LABELS_FILE = "seed_labels.json"

# Load dataset images and apply image processing 
def load_images(folder_path):
    images = []
    for i in range(1, 10001):
        img_path = os.path.join(folder_path, f"{i}.bmp")
        img = Image.open(img_path).convert('L') # grey scale
        # Increase image contrast 
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2)
        img_array = np.array(img).flatten() # 784-dim vector
        images.append(img_array)
    return np.array(images)  # shape: (10000, 784)

# def perpare_labels_json(seed_indices):
#     template = {str(int(idx)): None for idx in seed_indices}

#     with open("seed_labels.json", "w") as f:
#         json.dump(template, f, indent=4)

# Apply data augmentation
def augment_image(flat_img):
    img = flat_img.reshape(28, 28)
    augmented = []
    augmented.append(rotate(img, 5, reshape=False).flatten())   # Rotate (+5 degrees)
    augmented.append(rotate(img, -5, reshape=False).flatten())  # Rotate (-5 degrees)
    augmented.append((img + np.random.normal(0, 10, img.shape)).clip(0,255).flatten())  # Gaussian noise
    augmented.append(shift(img, [2, 0]).flatten())  # Shift down
    augmented.append(shift(img, [-2, 0]).flatten()) # Shift up
    augmented.append(shift(img, [0, 2]).flatten())  # Shift right
    augmented.append(shift(img, [0, -2]).flatten()) # Shift left
    return augmented

# Apply feature extraction
def extract_hog_features(images_flat):
    features = []
    for img_flat in images_flat:
        img_2d = img_flat.reshape(28, 28)
        feat = hog(img_2d, pixels_per_cell=(7, 7),
                   cells_per_block=(2, 2), feature_vector=True)
        features.append(feat)
    return np.array(features)

# Get the indices of the n most ambiguous images
def get_boundary_images(svm, features, n=20):
     
    decision_scores = svm.decision_function(features)  # shape (10000, n_classes)
    
    # Sort scores per image and compute margin = top - second
    sorted_scores = np.sort(decision_scores, axis=1)
    margin = sorted_scores[:, -1] - sorted_scores[:, -2]
    
    # Lowest margin = most ambiguous
    boundary_idx = np.argsort(margin)[:n]
    return boundary_idx, margin

# Manually label the boundary images
def label_boundary_images(boundary_idx, images):

    human_labels = {}
    for idx in boundary_idx:
        plt.imshow(images[idx].reshape(28, 28), cmap='gray')
        plt.title(f"Image {idx+1}.bmp")
        plt.axis('off')
        plt.show(block=False)
        plt.pause(2)
        plt.close()
        while True:
            raw_label = input(f"Label for image {idx+1}: ").strip()
            if raw_label == "":
                print("Please enter a digit from 0 to 9, or -1 for mixed.")
                continue
            try:
                label = int(raw_label)
                break
            except ValueError:
                print("Invalid input. Enter a digit from 0 to 9, or -1 for mixed.")
        human_labels[idx] = label
    return human_labels

images = load_images("Indian_Digits_Train")
print(images.shape)  # (10000, 784)

np.random.seed(42)
seed_indices = np.random.choice(10000, 300, replace=False)

# Get the labels for the manually labelled 300 images 
with open("seed_labels.json") as f:
    labels_dict = json.load(f)

seed_labels = np.array([labels_dict[str(int(i))] for i in seed_indices])
print("Seed labels loaded")

# Prepare the augmented images
aug_images = []
aug_labels = []
for idx, lbl in zip(seed_indices, seed_labels):
    for aug in augment_image(images[idx]):
        aug_images.append(aug)
        aug_labels.append(lbl)

aug_images = np.array(aug_images)
aug_labels = np.array(aug_labels)
print("Agumentation done")

# Extract features 
all_features = extract_hog_features(images)  # shape: (10000, 324)
seed_features = all_features[seed_indices]
aug_features  = extract_hog_features(aug_images)
print("Feature extraction done")

# Combine seed images with weight=100 and augmented images with weight=1
X_train = np.vstack([seed_features, aug_features])
y_train = np.concatenate([seed_labels, aug_labels])
w_train = np.array([100]*len(seed_labels) + [1]*len(aug_labels))

# Train initial SVM
svm = SVC(kernel='rbf', decision_function_shape='ovo', probability=True)
svm.fit(X_train, y_train, sample_weight=w_train)

print("SVM-1 trained")

# Evaluate initial SVM 
all_preds = svm.predict(all_features).astype(int)
acc, n_correct, n_total = check_accuracy(all_preds)
print(f"\nInitial accuracy: {acc:.2%}  ({n_correct}/{n_total})")

# Manual effort tracking
total_images_labelled = 300 
total_time_seconds    = 300 * 10  # 10s per image

# Refinement
MAX_ITERATIONS = 20
num_boundary_images = 20 # number of boundary images to label per iteration

# Keep track of the used indices 
used_indices = set(seed_indices.tolist())

for iteration in range(1, MAX_ITERATIONS + 1):

    acc_old = acc

    print(f"\n{'='*40}")
    print(f"Iteration {iteration}")
    print(f"{'='*40}")

    # Find the most ambiguous images
    boundary_idx, margins = get_boundary_images(svm, all_features)
    boundary_images = all_features[boundary_idx]

    # Manually label the most ambiguous images
    human_labels = label_boundary_images(boundary_idx, images)
    boundary_labels = np.array([human_labels[i] for i in boundary_idx])

    # Update tarining arrays
    X_train = np.vstack([X_train, boundary_images])
    y_train = np.concatenate([y_train, boundary_labels])
    w_train = np.concatenate([w_train, [100] * len(boundary_idx)])

    used_indices.update(boundary_idx)

    # Apply self-training with the highest confidence as pseudo labels
    pred_labels = svm.predict(all_features)
    threshold   = np.percentile(margins, 75)

    pseudo_labels_added   = 0
    pseudo_labels_rejected = 0

    for cls in range(10):
        # Candidates pseudo labels 
        candidates = [
            (margins[i], i)
            for i in range(10000)
            if pred_labels[i] == cls
            and i not in used_indices
            and margins[i] > threshold
        ]

        # Sort by margin descendingly and choose top 50 to be used for training
        candidates.sort(key=lambda x: -x[0])
        top50 = candidates[:50]
        rejected = max(0, len(candidates) - 50)
        pseudo_labels_rejected += rejected

        for _, i in top50:
            X_train = np.vstack([X_train, all_features[i:i+1]])
            y_train = np.concatenate([y_train, [pred_labels[i]]])
            w_train = np.concatenate([w_train, [1]])
            used_indices.add(i)
            pseudo_labels_added += 1

    print(f"Pseudo labels added: {pseudo_labels_added}")
    print(f"Pseudo labels rejected: {pseudo_labels_rejected}")

    # Retrain 
    svm.fit(X_train, y_train, sample_weight=w_train)

    print(f"SVM-{iteration} trained")

    # Evaluate after every iteration
    all_preds = svm.predict(all_features).astype(int)
    acc, n_correct, n_total = check_accuracy(all_preds)
    print(f"Accuracy after iteration {iteration}: {acc:.2%}  ({n_correct}/{n_total})")

    # Keep track of manual effort
    total_images_labelled += num_boundary_images
    total_time_seconds    += num_boundary_images * 10  # 10s per image

    # Stop if target reached or accuracy saturates 
    if acc >= 0.99 or abs(acc - acc_old) < 0.0001:
        print(f"\nTarget reached at iteration {iteration}!")
        break

# Report
print(f"\n{'='*40}")
print(f"FINAL REPORT")
print(f"{'='*40}")
print(f"Final accuracy      : {acc:.2%}")
print(f"Iterations done     : {iteration + 1}")
print(f"Images labelled     : {total_images_labelled}")
print(f"Total manual time   : {total_time_seconds}s = {total_time_seconds/3600:.2f} hours")
print(f"vs baseline (27.8h) : saved {27.8 - total_time_seconds/3600:.1f} hours")

# Save the trained model
joblib.dump(svm, 'model_pipeline2.pkl')
print("Model saved!")
