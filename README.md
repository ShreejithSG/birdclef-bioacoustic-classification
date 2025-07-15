# BirdCLEF 2025 - Bioacoustic Bird Detector 🐦

This repository contains the core codebase developed for the BirdCLEF 2025 challenge, featuring a **novel per-species binary classification framework** for bioacoustic bird detection. The approach deviates from conventional multi-class CNN pipelines by building **dedicated binary classifiers for each species**, making it highly interpretable, modular, and scalable across hundreds of classes.

📢 Conference Publication & Presentation
Our final methodology was submitted as a working notes paper to CLEF 2025 has been accepted. CLEF 2025 (Conference and Labs of the Evaluation Forum) will take place 9–12 September 2025 in Madrid, Spain, hosted by UNED. The working notes, along with the lab overview, will be published in the CEUR Workshop Proceedings (CEUR‑WS) series

> 🧪 This project was developed as part of **DSC 291: Machine Learning Competitions** at UC San Diego, under the guidance of Prof. Berk Ustun.

---

## 📁 Repository Structure

| Notebook                           | Description                                                        |
|-----------------------------------|--------------------------------------------------------------------|
| `single_bird_detector_grekis.ipynb` | Binary classifier using frequency-bin features for species `grekis` |
| `3_bird_detector.ipynb`            | 1-vs-rest classifiers for 3 frequent species (grekis, compau, trokin) |
| `206-bird-training-final.ipynb`    | Full pipeline training 206 per-species models using metadata + augmentation |
| `inference.ipynb`                  | Chunk-wise inference pipeline across all 206 models for test soundscapes |

---

## 📂 Data

The official dataset is available on the [BirdCLEF 2025 Kaggle Competition](https://www.kaggle.com/competitions/birdclef-2025/data). Raw `.ogg` audio files (~12GB) are not included in this repo due to storage limits.

**Included metadata and artifacts:**
- `train.csv`: Original metadata
- `train_augmented.csv`: Metadata with synthetic data (for 31 low-F1 species)
- `leaderboard.json`: CV metrics for all 206 species
- `leaderboard_31.json`: Scores for 31 augmented species
- `models.zip`: Trained per-species XGBoost models (`.joblib`) for both original and augmented runs

---

## 🛠 Features and Highlights

- ✅ **Binary Frequency Binning**: Uses adaptive percentile thresholds on STFT energy bands to build 3200-dim binary feature vectors.
- 🐦 **Per-Species Binary Models**: One-vs-rest XGBoost classifiers for each bird species, trained independently.
- ⚖️ **Class Imbalance Handling**: Uses `scale_pos_weight`, secondary label weighting, and strategic under-sampling.
- 🗺️ **Metadata-Enhanced Learning**: Embeds location (lat/lon) and collection source (CSA/XC/iNat) into each feature vector.
- 📉 **Evaluation**: 5-fold stratified cross-validation logging both AUC and F1.
- 🧪 **Inference Pipeline**: Segments soundscapes into 5s chunks, runs predictions using 206 species-specific models.

---

## 🔁 Data Augmentation Strategy

To boost recall for underrepresented species (those with near-zero F1), ~2,000 new training samples were created using:

- `ts_0.9`, `ts_1.1` → Time-stretching
- `ps_up`, `ps_down` → Pitch shifting
- `gain` → Amplitude scaling

Augmented recordings were appended with updated filenames and reused the original feature extractor, requiring **no architectural changes**.

---

## 📝 Working Notes Paper

Our final methodology was submitted as a working notes paper to CLEF 2025 has been accepted. It will be published in the CEUR Workshop Proceedings (CEUR‑WS) series

📎 [`clef2025_working_note.pdf`](./clef2025_working_note.pdf)

---

## 🔧 Dependencies

- Python ≥ 3.8
- `xgboost`
- `librosa`
- `numpy`, `pandas`, `scikit-learn`
- `joblib`, `tqdm`, `matplotlib` (optional for plots)

Install dependencies with:

```bash
pip install -r requirements.txt
```

Special thanks to the BirdCLEF organizers, Halıcıoğlu Data Science Institute – UC San Diego, and our course advisor Prof. Berk Ustun for supporting the competition-driven research framework.
