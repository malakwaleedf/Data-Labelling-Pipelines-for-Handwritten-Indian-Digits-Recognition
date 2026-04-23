import json

# ── Config ────────────────────────────────────────────────────────────────────
AGREED_FILE  = "agreed_labels.txt"
PREDICT_FILE = "500_labels_for_testing.json"
# ─────────────────────────────────────────────────────────────────────────────

# Load agreed (ground truth) labels  {image_id: label}
agreed = {}
with open(AGREED_FILE) as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            agreed[int(parts[0])] = int(parts[1])

# Load model predictions  {image_id: label}
with open(PREDICT_FILE) as f:
    raw = json.load(f)
predictions = {int(k): int(v) for k, v in raw.items()}

# ── Overlap ───────────────────────────────────────────────────────────────────
overlap = sorted(set(agreed) & set(predictions))

# ── Accuracy ──────────────────────────────────────────────────────────────────
per_class_correct = {i: 0 for i in range(10)}
per_class_total   = {i: 0 for i in range(10)}
misclassified = []

for img_id in overlap:
    gt   = agreed[img_id]
    pred = predictions[img_id]
    per_class_total[gt] += 1
    if pred == gt:
        per_class_correct[gt] += 1
    else:
        misclassified.append((img_id, gt, pred))

correct  = sum(per_class_correct.values())
total    = len(overlap)
accuracy = correct / total * 100
missing_images = len(predictions) - len(overlap)

print("Final labelling accuracy aganist the 500 manually labelled images")
print(f"Final accuracy: {correct}/{total} = {accuracy:.2f}%")
print(f"Number missing images  = {missing_images} (2 LLMs disagree upon)")
