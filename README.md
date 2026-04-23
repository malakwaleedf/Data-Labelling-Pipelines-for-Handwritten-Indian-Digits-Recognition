# Data Labelling Pipelines for Handwritten Indian Digits Recognition

## Overview
Manual data labelling is expensive and time-consuming. This project investigates multiple semi-automated labelling pipelines designed to minimize human effort while maintaining high accuracy.

We work with a dataset of 10,000 unlabeled grayscale images (28×28) representing handwritten Indian digits (0–9). The goal is to assign correct labels using efficient pipelines with minimal manual intervention.

---

## Objectives
- Design and implement automated labelling pipelines
- Achieve ≥98% labelling accuracy
- Minimize total manual annotation time
- Compare different strategies in terms of accuracy and efficiency

---

## Pipelines Implemented

### 1. K-Means + Active SVM Refinement
- Feature extraction (HOG)
- K-means clustering (K = 60)
- Cluster-level human labelling
- SVM training
- Active learning on low-confidence samples
- Iterative refinement

---

### 2️2. Seed Labelling + Augmentation + Self-Training
- Manual labelling of 300 samples
- Data augmentation (rotation, noise, shifts)
- Initial SVM training
- Active learning (boundary samples)
- Self-training with high-confidence predictions
- Iterative retraining

---

### 3️⃣ LLM-Based Labelling with Agreement Validation
- Benchmark multiple vision-capable LLMs
- Select top 2 performing models
- Label dataset independently
- Accept agreed predictions
- Resolve disagreements manually

---
