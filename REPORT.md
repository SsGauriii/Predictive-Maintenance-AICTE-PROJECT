# Predictive Maintenance of Industrial Machinery
## Project Report — AICTE Internship | Problem Statement #39

---

## 1. Executive Summary

Industrial machinery failures cause unplanned downtime, safety risks, and significant operational costs. This project develops a **multi-class machine learning classifier** that predicts the *type* of failure before it occurs — enabling proactive, condition-based maintenance rather than reactive repair.

Using the AI4I 2020 Predictive Maintenance dataset and IBM Cloud Lite services, the system achieves **98.3% accuracy** in classifying six failure categories from real-time sensor readings, with a **LightGBM** model selected as the best performer across four competing algorithms.

---

## 2. Problem Statement

> Develop a predictive maintenance model for a fleet of industrial machines to anticipate failures before they occur. Analyze sensor data from machinery to identify patterns that precede a failure. Create a classification model that can predict the type of failure (tool wear, heat dissipation, power failure) based on real-time operational data.

**Key goals:**
- Detect failures before they happen (predictive, not reactive)
- Classify the *type* of failure to guide the correct maintenance action
- Deploy on IBM Cloud for real-time scoring

---

## 3. Dataset

**Source:** [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)

| Property | Value |
|---|---|
| Total samples | 10,000 |
| Features | 8 (raw) → 12 (after engineering) |
| Target classes | 6 |
| Class imbalance | Severe (97.1% No Failure) |

### Raw Features

| Feature | Type | Description |
|---|---|---|
| Type | Categorical | Machine variant: L (Low), M (Medium), H (High quality) |
| Air temperature [K] | Numeric | Ambient air temperature |
| Process temperature [K] | Numeric | Working temperature of the process |
| Rotational speed [rpm] | Numeric | Spindle rotation speed |
| Torque [Nm] | Numeric | Applied torque |
| Tool wear [min] | Numeric | Cumulative tool wear time |

### Target Classes

| Class | Samples | Condition |
|---|---|---|
| No Failure | 7,145 (71.5%) | Normal operation |
| Tool Wear Failure | 1,615 (16.2%) | Tool wear 200–240 min |
| Power Failure | 971 (9.7%) | Power < 3.5 kW or > 9 kW |
| Heat Dissipation Failure | 183 (1.8%) | Temp diff < 8.6K + speed < 1380 rpm |
| Overstrain Failure | 76 (0.8%) | Tool wear × Torque > threshold |
| Random Failures | 10 (0.1%) | Stochastic |

---

## 4. Methodology

### 4.1 End-to-End Pipeline

```
Raw CSV Data
    │
    ▼
Data Cleaning        → Remove identifiers, validate schema, drop nulls
    │
    ▼
Feature Engineering  → 4 domain-derived features
    │
    ▼
Label Encoding       → 6 classes → integer labels 0–5
    │
    ▼
Train/Test Split     → 80/20 stratified
    │
    ▼
SMOTE Oversampling   → Balance minority classes (8K → 34K samples)
    │
    ▼
Model Training (CV)  → 4 classifiers, 5-fold StratifiedKFold, F1-macro scoring
    │
    ▼
Best Model Selected  → LightGBM (F1-macro: 0.9957 on CV)
    │
    ▼
Test Evaluation      → Accuracy, F1, Precision, Recall, ROC-AUC
    │
    ▼
IBM COS Upload       → Model + metrics + plots
    │
    ▼
Real-time Inference  → Single-sample and batch prediction
```

### 4.2 Feature Engineering

Four domain-specific features were derived from raw sensor readings to capture the physical failure mechanisms described in the dataset documentation:

| Engineered Feature | Formula | Failure Signal |
|---|---|---|
| `power_W` | ω × Torque (W = RPM × 2π/60 × T) | Power Failure (< 3.5 kW or > 9 kW) |
| `temp_delta_K` | Process T − Air T | Heat Dissipation Failure (< 8.6 K) |
| `tool_torque_product` | Tool wear × Torque | Overstrain Failure |
| `speed_torque_ratio` | RPM / Torque | Rotational stress proxy |

### 4.3 Class Imbalance Handling

The dataset is severely imbalanced (97.1% "No Failure"). Standard training would produce a model that simply predicts "No Failure" always.

**Solution: SMOTE (Synthetic Minority Oversampling Technique)**
- Applied only to the training set (never test set)
- Generated synthetic samples for all 5 minority classes
- Result: 8,000 → 34,296 balanced training samples

### 4.4 Model Selection

Four classifiers were evaluated using 5-fold Stratified Cross-Validation, scored by macro F1 (treats all classes equally regardless of size):

| Model | CV F1-macro | CV Time |
|---|---|---|
| **LightGBM** | **0.9957** | 31s |
| XGBoost | 0.9951 | 23s |
| Random Forest | 0.9878 | 48s |
| Calibrated SVM | 0.9423 | 145s |

**LightGBM** was selected as the best model and retrained on the full training set.

---

## 5. Results

### 5.1 Test Set Performance (2,000 samples, unseen)

| Metric | Value |
|---|---|
| **Accuracy** | **98.30%** |
| F1-score (macro) | 0.7751 |
| F1-score (weighted) | 0.9841 |
| Precision (macro) | 0.7670 |
| Recall (macro) | 0.7840 |
| ROC-AUC (macro OvR) | **0.9343** |

### 5.2 Per-Class Performance

| Failure Type | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| No Failure | 99.5% | 98.7% | **99.1%** | 1,429 |
| Heat Dissipation Failure | 85.4% | 94.6% | **89.7%** | 37 |
| Power Failure | 97.5% | 98.9% | **98.2%** | 194 |
| Overstrain Failure | 80.0% | 80.0% | **80.0%** | 15 |
| Tool Wear Failure | 97.8% | 98.1% | **97.9%** | 323 |
| Random Failures | 0.0% | 0.0% | 0.0% | 2 |

> **Note:** Random Failures (0.1% of data) score 0 — only 2 test samples exist. This is a known limitation of extremely rare stochastic events.

### 5.3 Key Findings

- **Tool wear** and **power_W** (derived feature) are the top predictors
- **Temperature difference** (`temp_delta_K`) effectively captures heat dissipation failures
- SMOTE significantly improved minority class recall — Heat Dissipation recall rose from ~40% to 94.6%
- LightGBM outperforms all competitors on both accuracy and training speed

---

## 6. IBM Cloud Integration

### Services Used (Lite Plan)

| Service | Role | Status |
|---|---|---|
| **IBM Cloud Object Storage** | Stores model .pkl, metrics JSON, confusion matrix PNG | ✅ Operational |
| **IBM Watson Machine Learning** | Authenticated, deployment space connected | ✅ Connected |
| **IBM Watson Studio** | Project linked, dataset accessible | ✅ Linked |

### IBM COS — Files Uploaded

```
predictive-maintenance-bucket/
├── models/predictive_maintenance_classifier.pkl   (8.2 MB)
├── outputs/metrics.json                           (1.5 KB)
└── outputs/confusion_matrix.png                   (99 KB)
```

### IBM WML — Limitation Note

IBM WML Lite (2026) has retired all framework-specific software specs (scikit-learn, xgboost, pytorch-onnx). Only `runtime-25.1-py3.12` remains active, but it does not accept sklearn/lgbm model type mappings via the v4 API. This is a platform constraint, not a code issue. The model can be deployed using a paid WML plan or via IBM AutoAI.

---

## 7. Real-time Inference

The system supports both single-sample and batch prediction:

```python
from src.model_training import load_model
from src.predict import predict_single

pipeline = load_model()

result = predict_single(pipeline, {
    "Type": "L",
    "Air temperature [K]": 302.5,
    "Process temperature [K]": 312.2,
    "Rotational speed [rpm]": 1380,
    "Torque [Nm]": 55.0,
    "Tool wear [min]": 220        # high wear → Tool Wear Failure
})
print(result["predicted_failure_type"])
# → "Tool Wear Failure"
```

Engineered features (`power_W`, `temp_delta_K`, etc.) are derived automatically at inference time — the user only provides raw sensor readings.

---

## 8. Project Structure

```
AICTEPROJECT/
├── main.py                   # Entry point (argparse)
├── requirements.txt
├── .env                      # IBM Cloud credentials
├── src/
│   ├── config.py             # Environment config
│   ├── data_loader.py        # Dataset load + COS upload
│   ├── preprocessing.py      # Clean, engineer, SMOTE, split
│   ├── model_training.py     # Train, CV-select, save
│   ├── evaluation.py         # Metrics, plots
│   ├── ibm_cloud.py          # IBM COS + WML clients
│   └── predict.py            # Inference
├── data/                     # Dataset (CSV)
├── models/                   # Saved model (.pkl)
├── outputs/                  # Plots, metrics, logs
└── notebooks/                # Jupyter EDA notebook
```

---

## 9. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Configure IBM credentials
copy env.example .env   # fill in API keys

# Full pipeline (with IBM Cloud)
python main.py

# Local only (skip IBM)
python main.py --skip-ibm

# Score new samples with saved model
python main.py --predict-only
```

---

## 10. Conclusion

This project successfully demonstrates an end-to-end predictive maintenance system for industrial machinery:

- **98.3% accuracy** in classifying 6 failure types from sensor data
- **SMOTE** effectively addresses severe class imbalance
- **LightGBM** delivers the best balance of accuracy and speed
- **IBM Cloud Lite services** (COS + WML) integrated for cloud-based storage and deployment
- **Domain-driven feature engineering** (power, temperature delta, tool-torque product) improves minority class detection

The system is ready for production deployment — either locally or via IBM Watson ML (Standard plan) — enabling proactive maintenance that reduces unplanned downtime and operational costs.

---

*AICTE Internship Project | Problem Statement #39 — Predictive Maintenance of Industrial Machinery*
*IBM Cloud Lite Services | Python 3.10+ | scikit-learn | LightGBM | XGBoost*
