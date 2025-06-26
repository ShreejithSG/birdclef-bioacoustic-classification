# BirdCLEF 2025 - Bioacoustic Bird Detector 🐦

This repository contains the final training and inference pipelines developed for the BirdCLEF 2025 competition, centered around a novel **per-species binary classification framework** for bird audio detection.

## 📌 Files

| Notebook                        | Description                                      |
|--------------------------------|--------------------------------------------------|
| `single_bird_detector_grekis.ipynb` | Initial binary classifier for top species       |
| `3_bird detector.ipynb`  | Prototype with 1-vs-rest classifiers (3 birds)  |
| `206-bird-training-final.ipynb`| Final pipeline training 206 per-species models |
| `inference.ipynb`   | Inference and Kaggle submission generation      |

## 🛠 Features
- Binary frequency binning via adaptive thresholds
- Per-species model training with class weighting
- Stratified CV evaluation with AUC + F1
- Custom inference system for soundscape chunks

## 📝 Status
🧪 Submitted as a working notes paper to CLEF 2025 (under review).
