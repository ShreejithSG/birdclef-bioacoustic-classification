# BirdCLEF 2025 - Bioacoustic Bird Detector 🐦

This repository contains the core codebase developed for the BirdCLEF 2025 challenge, featuring a **novel per-species binary classification framework** for bioacoustic bird detection. The approach deviates from conventional multi-class CNN pipelines by building **dedicated binary classifiers for each species**, making it highly interpretable, modular, and easier to scale across hundreds of species.

The final methodology was submitted as a working notes paper to **CLEF 2025** and is currently under review.

---

## 📁 Repository Structure

| Notebook                           | Description                                                        |
|-----------------------------------|--------------------------------------------------------------------|
| `single_bird_detector_grekis.ipynb` | Binary classifier using frequency-bin features for species `grekis` |
| `3_bird_detector.ipynb`            | 1-vs-rest classifiers for 3 most frequent species (grekis, compau, trokin) |
| `206-bird-training-final.ipynb`    | Full training pipeline: 206 per-species models with metadata + augmentation |
| `inference.ipynb`                  | Custom inference pipeline over soundscape chunks for all 206 models |

---

## 🛠 Features and Highlights

- ✅ **Binary Frequency Binning**: Adaptive thresholding over short-time Fourier transform (STFT) outputs to construct compact 3200-dim binary feature vectors.
- 🎯 **Per-Species Classifiers**: One model per bird species trained in parallel using XGBoost, enabling interpretable one-vs-rest decision boundaries.
- ⚖️ **Class Imbalance Handling**: Dynamic `scale_pos_weight`, secondary label upweighting, and optional data augmentation (pitch, speed, time-shift, gain).
- 📍 **Metadata Integration**: Longitude, latitude, and recording collection (CSA/XC/iNat) encoded for improved context-aware predictions.
- 📉 **Cross-Validation**: Stratified 5-fold CV with AUC and F1 score logging for leaderboard generation.
- 🧪 **Test-Time Inference**: Chunk-wise prediction with model confidence thresholding across test soundscapes.

## 🔁 Data Augmentation Strategy

To mitigate poor performance for underrepresented classes, we augmented ~2,000 training clips across 31 species using:

- `ts_0.9`, `ts_1.1`: Time-stretching
- `ps_up`, `ps_down`: Pitch shifts
- `gain`: Volume amplification

Each augmented file was appended to the training set with updated metadata. No architectural changes were required, and the final training pipeline supports both original and augmented samples seamlessly.

---

## 📝 Paper Status

This work has been **submitted as a working notes paper to CLEF 2025** under the LifeCLEF Bird Subtask and is currently under review.

📎 `clef2025_working_note.pdf` – [Submitted version included in this repo](./clef2025_working_note.pdf)

---

## 🔧 Dependencies

- Python ≥ 3.8
- `xgboost`
- `librosa`
- `numpy`, `pandas`, `scikit-learn`
- `joblib`, `tqdm`, `matplotlib` (optional for visualizations)

Install using:
```bash
pip install -r requirements.txt
