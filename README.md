# Industrial Digital Twin: Predictive Maintenance System
**Stacking Ensemble & Physics-Informed ML**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=for-the-badge&logo=flask&logoColor=white)
![ML](https://img.shields.io/badge/Sklearn-Stacking-orange?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research_Prototype-red?style=for-the-badge)

##  Project Overview

This project is an **Industrial Digital Twin** designed to predict machine failures before they occur. Unlike standard "black box" classifiers, this system leverages **Physics-Informed Machine Learning** combining raw sensor data with domain-specific physical formulas (Strain, Heat Dissipation, Power Factor) to assess equipment health.

The core is a **Stacking Ensemble Classifier** (XGBoost + Random Forest + Logistic Regression Meta-Learner) that achieves high recall on rare failure events. The results are deployed via a **Flask Web Dashboard** that provides real-time probabilistic risk assessment and interpretable diagnostics (XAI).

---

##  Key Features

### Physics-Informed Feature Engineering
Synthesizes physical metrics like **Heat Dissipation** (`Process Temp - Air Temp`) and **Tool Strain** (`Torque × Tool Wear`) to capture actual mechanical stress beyond raw sensor readings.

### Stacking Ensemble Architecture
* **Level 0:** XGBoost (Gradient Boosting) & Random Forest (Bagging) for robust feature extraction.
* **Level 1:** Logistic Regression Meta-Learner to minimize variance and bias, optimizing the decision boundary.

### Class Imbalance Handling
Uses **SMOTE (Synthetic Minority Over-sampling Technique)** within a stratified cross-validation pipeline to robustly detect rare failure events (failures represent only ~3% of the dataset).

### Calibrated Risk Probability
Outputs a continuous **"Failure Probability" score (0-100%)** calibrated via Isotonic Regression. This moves beyond simple binary classification to provide actionable risk gradients.

### Digital Twin Dashboard
Visualizes the machine's state via a dynamic **Radar Chart**, allowing operators to detect multidimensional anomalies (e.g., specific correlations between High Torque and High Wear).

---

## Tech Stack

| Domain | Technologies |
| :--- | :--- |
| **Language** | Python 3.x |
| **Web Framework** | Flask (Jinja2 Templates) |
| **Machine Learning** | Scikit-Learn, XGBoost, Imbalanced-Learn |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib (Server-side rendering), Chart.js |
| **Serialization** | Joblib |

---

## 📂 Project Structure

```text
PredictiveMaintenance_Pro/
│
├── data/                  # Dataset (UCI AI4I 2020)
├── models/                # Trained Artifacts (.pkl files)
├── src/                   # Research Lab (Training Pipeline)
│   ├── train_pipeline.py  # Main Stacking Ensemble Script
│   └── ...
├── web_app/               # Deployment (Flask)
│   ├── static/            # CSS/Assets
│   ├── templates/         # HTML Dashboard
│   └── app.py             # Flask Server Logic
├── requirements.txt       # Dependencies
└── README.md              # Documentation