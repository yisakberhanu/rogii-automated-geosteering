# Automated Geosteering: Subsurface Sequence Alignment

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Geospatial-orange)
![Status](https://img.shields.io/badge/Status-Active_Development-success)

## 📌 Project Overview
This repository contains my machine learning pipeline for the **ROGII - Wellbore Geology Prediction** challenge. The objective is to automate geosteering operations by predicting the geology (True Vertical Thickness, or TVT) encountered along a horizontal wellbore using standard drilling sensor data (Gamma Ray).

Accurate subsurface mapping reduces resource waste, minimizes environmental footprint, and improves drilling safety. 

## 🧠 The Core Challenge
This is not a standard tabular regression problem; it is a **dynamic spatial mapping problem**. 
* **The Query:** Horizontal well telemetry (Measured Depth, X, Y, Z, and Gamma Ray).
* **The Reference:** A vertical "Typewell" containing the true geological layer signatures.
* **The Goal:** Align the horizontal sequence to the vertical sequence to predict the exact TVT in a hidden evaluation zone.

## 🛠️ Technical Strategy & Highlights
To ensure robust, production-ready predictions, this pipeline is built with several strict data science principles:

1. **Dynamic Evaluation Zone Detection:** The prediction horizon varies for every single well (e.g., predicting 500 rows vs 5,000 rows into the future). The pipeline dynamically detects the start of the `NaN` evaluation zone to establish the final known anchor point.
2. **Strict Data Leakage Prevention (GroupKFold):** Because wellbore sensor readings are highly autocorrelated, random train/test splits result in catastrophic data leakage. My validation strategy utilizes a strict `GroupKFold` split grouped by `WELL_ID`, ensuring the model is evaluated exactly as it will be in the real world.
3. **Signal Interpolation:** Real-world sensor telemetry contains dropouts. The data ingestion layer uses linear interpolation to handle missing Gamma Ray (`GR`) readings before feature engineering begins.

## 📂 Repository Structure
```text
├── data/                   # (Ignored in git) Kaggle dataset files
├── notebooks/              # Jupyter notebooks for EDA and visualization
├── src/                    # Source code for the main pipeline
│   ├── data_ingestion.py   # Data loading and dynamic zone detection
│   ├── validation.py       # GroupKFold generation logic
│   └── baseline_model.py   # Baseline prediction algorithms
└── README.md
