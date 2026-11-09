# Predictive Maintenance of Industrial Machinery
### AICTE Project – Problem Statement #39

> **Goal:** Classify the type of industrial machine failure (Tool Wear, Heat Dissipation, Power Failure, Overstrain, Random, No Failure) from real-time sensor data using IBM Cloud Lite services.

---

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) from Kaggle.

| Column | Description |
|--------|-------------|
| Type | Machine product type (L / M / H) |
| Air temperature [K] | Air temperature in Kelvin |
| Process temperature [K] | Process temperature in Kelvin |
| Rotational speed [rpm] | Spindle speed |
| Torque [Nm] | Applied torque |
| Tool wear [min] | Cumulative tool wear |
| Failure Type | **Target** – one of 6 classes |

Place the downloaded CSV at `data/predictive_maintenance.csv`.  
If absent, the pipeline auto-generates a synthetic dataset for validation.

---

## Architecture

```
Raw Sensor Data
      │
      ▼
┌─────────────────────────────────┐
│  Data Loader (data_loader.py)   │  ← Local CSV or synthetic data
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Preprocessing (preprocessing.py)│  ← Clean, engineer features, SMOTE
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Model Training (model_training)│  ← RF · XGBoost · LightGBM · SVM
│  CV selection → best model      │    StratifiedKFold · F1-macro
└──────────────┬──────────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌──────────────┐  ┌────────────────────────────────┐
│ Local Model  │  │   IBM Cloud Lite Services       │
│ models/*.pkl │  │  ┌──────────────────────────┐   │
└──────────────┘  │  │ Watson Machine Learning  │   │
                  │  │  (store + online deploy) │   │
                  │  └──────────────────────────┘   │
                  │  ┌──────────────────────────┐   │
                  │  │ Cloud Object Storage     │   │
                  │  │  (datasets + artefacts)  │   │
                  │  └──────────────────────────┘   │
                  └────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Evaluation (evaluation.py)     │  ← Confusion matrix, F1, ROC-AUC
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Inference (predict.py)         │  ← Single-sample & batch scoring
└─────────────────────────────────┘
```

---

## Project Structure

```
AICTEPROJECT/
├── main.py                    # End-to-end pipeline entry point
├── requirements.txt
├── env.example                # Copy to .env and fill credentials
├── .env                       # IBM Cloud credentials (git-ignored)
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Centralised config from .env
│   ├── data_loader.py         # Dataset loading + COS upload
│   ├── preprocessing.py       # Cleaning, feature engineering, SMOTE
│   ├── model_training.py      # Train, CV-select, persist model
│   ├── evaluation.py          # Metrics, confusion matrix, plots
│   ├── ibm_cloud.py           # Watson ML + COS integration
│   └── predict.py             # Inference (local + WML scoring)
│
├── data/
│   └── predictive_maintenance.csv   # Place Kaggle dataset here
│
├── models/
│   └── predictive_maintenance_classifier.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── cv_comparison.png
│   ├── class_distribution_*.png
│   ├── metrics.json
│   └── pipeline.log
│
└── notebooks/
    └── (optional Jupyter notebooks for exploration)
```

---

## IBM Cloud Lite Services Setup

### 1. Watson Machine Learning (Free tier)
1. Log in to [IBM Cloud](https://cloud.ibm.com).
2. Create a **Watson Machine Learning** service (Lite plan).
3. Create a **Deployment Space** (Manage → Deployment Spaces → New).
4. Go to **Manage → Access (IAM)** → Create API key.
5. Copy the **API Key**, **Instance ID**, and **Space ID** into `.env`.

### 2. Cloud Object Storage (Free tier)
1. Create a **Cloud Object Storage** instance (Lite plan).
2. Create a bucket named `predictive-maintenance-bucket` (US South region).
3. Go to **Service credentials** → New credential → include HMAC.
4. Copy **apikey** and **resource_instance_id** into `.env`.

---

## Quick Start

```bash
# 1. Clone / download the project
cd AICTEPROJECT

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure IBM Cloud credentials
copy env.example .env          # Windows
# cp env.example .env          # Linux/macOS
# Edit .env and fill in your IBM Cloud API keys

# 5. (Optional) Place Kaggle dataset
# Download from https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification
# Place CSV at:  data/predictive_maintenance.csv

# 6. Run the full pipeline
python main.py

# 7. Local-only run (skip IBM Cloud steps)
python main.py --skip-ibm

# 8. Score demo samples with a saved model
python main.py --predict-only
```

---

## Engineered Features

Beyond the raw sensor readings, the pipeline derives:

| Feature | Formula | Failure Signal |
|---------|---------|----------------|
| `power_W` | ω × Torque | Power failure (< 3.5 kW or > 9 kW) |
| `temp_delta_K` | Process T − Air T | Heat dissipation failure (< 8.6 K) |
| `tool_torque_product` | Tool wear × Torque | Overstrain failure |
| `speed_torque_ratio` | RPM / Torque | Rotational stress indicator |

---

## Failure Classes

| Label | Class | Typical Condition |
|-------|-------|-------------------|
| 0 | No Failure | Normal operation |
| 1 | Heat Dissipation Failure | Temp diff < 8.6 K + speed < 1380 rpm |
| 2 | Power Failure | Power < 3.5 kW or > 9 kW |
| 3 | Overstrain Failure | Tool wear × torque exceeds type threshold |
| 4 | Tool Wear Failure | Tool wear 200–240 min |
| 5 | Random Failures | Stochastic (0.1% probability) |

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/confusion_matrix.png` | Normalised confusion matrix |
| `outputs/feature_importance.png` | Top feature importances |
| `outputs/cv_comparison.png` | CV F1-macro comparison across classifiers |
| `outputs/class_distribution*.png` | Class balance before/after SMOTE |
| `outputs/metrics.json` | Full evaluation metrics (accuracy, F1, AUC) |
| `outputs/pipeline.log` | Full run log |
| `models/*.pkl` | Serialised best-model pipeline |

---

## Technology Stack

- **Python 3.10+**
- **scikit-learn** – preprocessing pipelines, SVM, Random Forest
- **XGBoost / LightGBM** – gradient boosted tree classifiers  
- **imbalanced-learn** – SMOTE oversampling
- **pandas / numpy** – data manipulation
- **matplotlib / seaborn** – visualisation
- **IBM Watson Machine Learning** – cloud model deployment (Lite tier)
- **IBM Cloud Object Storage** – artefact storage (Lite tier)
- **python-dotenv** – credential management

---

*AICTE Internship Project | Problem Statement #39 – Predictive Maintenance of Industrial Machinery*
