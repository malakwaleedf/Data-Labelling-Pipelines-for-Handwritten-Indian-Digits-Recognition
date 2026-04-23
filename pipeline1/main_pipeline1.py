import os
import random
import json
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from check_accuracy import check_accuracy
import joblib

CLUSTER_LABELS_FILE = "cluster_labels.json"

# Load dataset images and apply image processing 
def load_images(folder_path):
    images = []
    for i in range(1, 10001):
        img_path = os.path.join(folder_path, f"{i}.bmp")
        img = Image.open(img_path).convert('L') # grey scale
        # Increase image contrast 
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        img_array = np.array(img).flatten() # 784-dim vector
        images.append(img_array)
    return np.array(images)  # shape: (10000, 784)

# Apply feature extraction
def extract_hog_features(images_flat):
    features = []
    for img_flat in images_flat:
        img_2d = img_flat.reshape(28, 28)
        feat = hog(img_2d, pixels_per_cell=(7, 7),
                   cells_per_block=(2, 2), feature_vector=True)
        features.append(feat)
    return np.array(features)

# Show 8 sample images from a cluster for manual labelling 
def show_cluster_samples(cluster_id, cluster_assignments, images, n=8):

    indices = np.where(cluster_assignments == cluster_id)[0]
    sample_idx = random.sample(list(indices), min(n, len(indices)))
    
    fig, axes = plt.subplots(1, len(sample_idx), figsize=(12, 2))
    for ax, idx in zip(axes, sample_idx):
        ax.imshow(images[idx].reshape(28, 28), cmap='gray')
        ax.axis('off')
    plt.suptitle(f"Cluster {cluster_id}")
    plt.show(block=False)
    plt.pause(2)
    plt.close()

# Get the indices of the n most ambiguous images
def get_boundary_images(svm, features, n=30):
    
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
        plt.title(f"Image {idx}.bmp")
        plt.axis('off')
        plt.show(block=False)
        plt.pause(2)
        plt.close()
        while True:
            raw_label = input(f"Label for image {idx}: ").strip()
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

# Update tarining arrays and retrain 
def retrain_svm(X_train, y_train, w_train, human_labels, features):
    human_idx = list(human_labels.keys())
    human_y   = np.array([human_labels[i] for i in human_idx])
    human_X   = features[human_idx]
    human_w   = np.full(len(human_idx), 100.0)

    X_combined = np.vstack([X_train, human_X])
    y_combined = np.concatenate([y_train, human_y])
    w_combined = np.concatenate([w_train, human_w])

    svm_new = SVC(kernel='rbf', probability=True, decision_function_shape='ovo')
    svm_new.fit(X_combined, y_combined, sample_weight=w_combined)
    return svm_new, X_combined, y_combined, w_combined

def load_cluster_labels(path=CLUSTER_LABELS_FILE):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): int(v) for k, v in data.items()}

def save_cluster_labels(cluster_labels, path=CLUSTER_LABELS_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({int(k): int(v) for k, v in cluster_labels.items()}, f, indent=2)

images = load_images("Indian_Digits_Train")
print(images.shape)  # (10000, 784)

features = extract_hog_features(images)
print(features.shape)  # (10000, 324)

# Number of clusters 
K = 60

# Apply clustering
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans.fit(features)

cluster_assignments = kmeans.labels_ 
print(f"Cluster sizes: {np.bincount(cluster_assignments)}")

# Load saved clusters labels (if any)
cluster_labels = load_cluster_labels()

# Manually label each cluster with its corresponding correct digit
for k in range(K):
    if k in cluster_labels:
        continue
    show_cluster_samples(k, cluster_assignments, images)
    label = int(input(f"Cluster {k} digit label (0-9, or -1 if mixed): "))
    cluster_labels[k] = label

# Save clusters labels
save_cluster_labels(cluster_labels)

# Assign each image the label of its cluster
image_labels = np.array([cluster_labels[c] for c in cluster_assignments])
image_weights = np.ones(10000)  # weight = 1 for all cluster-assigned labels

# Remove mixed clusters elements them from training
valid_mask = image_labels != -1
X_train = features[valid_mask]
y_train = image_labels[valid_mask]
w_train = image_weights[valid_mask]

# Train initial SVM
svm = SVC(kernel='rbf', probability=True, decision_function_shape='ovo')
svm.fit(X_train, y_train, sample_weight=w_train)

print("Initial SVM trained!")

# Evaluate initial SVM 
all_preds = svm.predict(features).astype(int)
acc, n_correct, n_total = check_accuracy(all_preds)
print(f"\nInitial accuracy: {acc:.2%}  ({n_correct}/{n_total})")

# Manual effort tracking
total_images_labelled = K   
total_time_seconds = 20 * K # 20s per cluster

# Refinement
MAX_ITERATIONS = 20
num_boundary_images = 30 # number of boundary images to label per iteration

for iteration in range(1, MAX_ITERATIONS + 1):

    acc_old = acc

    print(f"\n{'='*40}")
    print(f"Iteration {iteration}")
    print(f"{'='*40}")

    # Find the most ambiguous images
    boundary_idx, _ = get_boundary_images(svm, features)

    # Manually label the most ambiguous images
    human_labels = label_boundary_images(boundary_idx, images)

    # Retrain
    svm, X_train, y_train, w_train = retrain_svm(
        X_train, y_train, w_train, human_labels, features
    )

    # Evaluate after every iteration
    all_preds = svm.predict(features).astype(int)
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
# joblib.dump(svm, 'model_pipeline1.pkl')
# print("Model saved!")
