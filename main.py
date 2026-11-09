"""
main.py
──────────────────────────────────────────────────────────────────────────────
Predictive Maintenance of Industrial Machinery
Problem Statement #39 – AICTE Project

End-to-end pipeline
    1. Load dataset (local CSV or synthetic fallback)
    2. Clean & engineer features
    3. Encode target labels
    4. Split data + apply SMOTE for class-balance
    5. Train & compare classifiers (CV-selected best model)
    6. Evaluate on hold-out test set
    7. Save model locally
    8. Upload artefacts to IBM Cloud Object Storage (Lite tier)
    9. Store & deploy model on IBM Watson Machine Learning (Lite tier)
   10. Demonstrate real-time inference

Usage
─────
    python main.py                    # full pipeline
    python main.py --skip-ibm         # skip IBM Cloud steps (local only)
    python main.py --predict-only     # load saved model + score demo samples
"""

import argparse
import json
import logging
import os
import sys
import time

import pandas as pd

from src.config import config
from src.data_loader import load_dataset
from src.preprocessing import (
    clean_data,
    engineer_features,
    encode_target,
    split_data,
    apply_smote,
)
from src.model_training import train_and_select, save_model, load_model
from src.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_cv_results,
    plot_class_distribution,
)
from src.predict import predict_single, predict_batch

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.OUTPUT_DIR, "pipeline.log"), mode="w"),
    ],
)
logger = logging.getLogger("main")

os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)


# ── Demo samples for real-time inference demo ─────────────────────────────────

DEMO_SAMPLES = [
    {   # Expected: No Failure
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
    },
    {   # Expected: Tool Wear Failure (high tool wear + medium torque)
        "Type": "L",
        "Air temperature [K]": 302.5,
        "Process temperature [K]": 312.2,
        "Rotational speed [rpm]": 1380,
        "Torque [Nm]": 55.0,
        "Tool wear [min]": 220,
    },
    {   # Expected: Heat Dissipation Failure (low temp diff + low speed)
        "Type": "H",
        "Air temperature [K]": 305.0,
        "Process temperature [K]": 311.5,
        "Rotational speed [rpm]": 1200,
        "Torque [Nm]": 30.0,
        "Tool wear [min]": 50,
    },
    {   # Expected: Power Failure (very low rotational speed → low power)
        "Type": "M",
        "Air temperature [K]": 299.0,
        "Process temperature [K]": 309.5,
        "Rotational speed [rpm]": 1168,
        "Torque [Nm]": 3.8,
        "Tool wear [min]": 10,
    },
    {   # Expected: Overstrain Failure (high tool wear × high torque)
        "Type": "L",
        "Air temperature [K]": 301.0,
        "Process temperature [K]": 311.0,
        "Rotational speed [rpm]": 1600,
        "Torque [Nm]": 76.0,
        "Tool wear [min]": 200,
    },
]


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(skip_ibm: bool = False) -> None:
    """Execute the full training + evaluation pipeline."""
    logger.info("=" * 70)
    logger.info("  Predictive Maintenance – Industrial Machinery (Problem #39)")
    logger.info("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("\n[STEP 1] Loading dataset …")
    df_raw = load_dataset()
    logger.info("Raw data shape: %s | Columns: %s", df_raw.shape, list(df_raw.columns))

    # ── 2. Clean + feature engineering ───────────────────────────────────────
    logger.info("\n[STEP 2] Cleaning and engineering features …")
    df_clean = clean_data(df_raw)
    df_feat = engineer_features(df_clean)

    # ── 3. Encode target ──────────────────────────────────────────────────────
    logger.info("\n[STEP 3] Encoding target labels …")
    X, y = encode_target(df_feat)
    logger.info("Feature matrix: %s | Classes: %s", X.shape, dict(zip(*pd.Series(y).value_counts().reset_index().values.T.tolist())))

    # Plot class distribution before SMOTE
    plot_class_distribution(y, title="Class Distribution (Before SMOTE)")

    # ── 4. Train / test split + SMOTE ─────────────────────────────────────────
    logger.info("\n[STEP 4] Splitting data and applying SMOTE …")
    numeric_features = [c for c in config.NUMERIC_FEATURES if c in X.columns]
    engineered_features = ["power_W", "temp_delta_K", "tool_torque_product", "speed_torque_ratio"]
    all_numeric = numeric_features + [f for f in engineered_features if f in X.columns]
    categorical_features = [c for c in config.CATEGORICAL_FEATURES if c in X.columns]

    X_train, X_test, y_train, y_test = split_data(X, y)

    # Encode 'Type' column to numeric before SMOTE (SMOTE requires all-numeric input)
    type_map = {"L": 0, "M": 1, "H": 2}
    X_train_num = X_train.copy()
    X_test_num  = X_test.copy()
    if "Type" in X_train_num.columns:
        X_train_num["Type"] = X_train_num["Type"].map(type_map).fillna(1)
        X_test_num["Type"]  = X_test_num["Type"].map(type_map).fillna(1)

    X_train_bal_arr, y_train_bal = apply_smote(X_train_num.values, y_train)
    X_train_bal = pd.DataFrame(X_train_bal_arr, columns=X_train_num.columns)

    # Decode numeric Type back to string for the sklearn OrdinalEncoder in the pipeline
    inv_type_map = {0: "L", 1: "M", 2: "H"}
    if "Type" in X_train_bal.columns:
        X_train_bal["Type"] = X_train_bal["Type"].round().astype(int).map(inv_type_map).fillna("M")

    # ── 5. Model training ─────────────────────────────────────────────────────
    logger.info("\n[STEP 5] Training and selecting best model …")

    best_pipeline, cv_results = train_and_select(
        X_train_bal,
        y_train_bal,
        numeric_features=all_numeric,
        categorical_features=categorical_features,
    )

    # ── 6. Evaluate on test set ───────────────────────────────────────────────
    logger.info("\n[STEP 6] Evaluating on hold-out test set …")
    metrics = evaluate_model(best_pipeline, X_test, y_test)

    plot_confusion_matrix(best_pipeline, X_test, y_test)
    plot_cv_results(cv_results)

    all_feature_names = all_numeric + categorical_features
    plot_feature_importance(best_pipeline, all_feature_names)

    # Persist metrics as JSON
    metrics_path = os.path.join(config.OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved to: %s", metrics_path)

    logger.info("\n-- Final Test-Set Metrics -----------------------------------------")
    logger.info("  Accuracy   : %.4f", metrics["accuracy"])
    logger.info("  F1 (macro) : %.4f", metrics["f1_macro"])
    logger.info("  F1 (wtd)   : %.4f", metrics["f1_weighted"])
    logger.info("  Prec (mac) : %.4f", metrics["precision_macro"])
    logger.info("  Recall(mac): %.4f", metrics["recall_macro"])
    if "roc_auc_macro" in metrics:
        logger.info("  ROC-AUC    : %.4f", metrics["roc_auc_macro"])

    # ── 7. Save model locally ─────────────────────────────────────────────────
    logger.info("\n[STEP 7] Saving model locally …")
    model_path = save_model(best_pipeline)

    # ── 8. IBM Cloud Object Storage upload ────────────────────────────────────
    if not skip_ibm:
        logger.info("\n[STEP 8] Uploading artefacts to IBM Cloud Object Storage …")
        _upload_artefacts_to_cos(model_path)

    # ── 9. IBM Watson Machine Learning deployment ─────────────────────────────
    if not skip_ibm:
        logger.info("\n[STEP 9] Storing and deploying model on IBM Watson ML …")
        _deploy_to_wml(best_pipeline)

    # ── 10. Real-time inference demo ──────────────────────────────────────────
    logger.info("\n[STEP 10] Real-time inference demo …")
    _run_inference_demo(best_pipeline)

    logger.info("\n%s", "=" * 70)
    logger.info("  Pipeline complete! Outputs saved to: %s/", config.OUTPUT_DIR)
    logger.info("=" * 70)


def _upload_artefacts_to_cos(model_path: str) -> None:
    from src.ibm_cloud import get_cos_client, upload_file_to_cos

    cos = get_cos_client()
    if cos is None:
        logger.warning("IBM COS client not available – skipping upload.")
        return

    artefacts = [
        (model_path, "models/" + os.path.basename(model_path)),
        (os.path.join(config.OUTPUT_DIR, "metrics.json"), "outputs/metrics.json"),
        (os.path.join(config.OUTPUT_DIR, "confusion_matrix.png"), "outputs/confusion_matrix.png"),
    ]
    for local_path, key in artefacts:
        if os.path.exists(local_path):
            upload_file_to_cos(local_path, object_key=key, cos_client=cos)


def _deploy_to_wml(pipeline) -> None:
    from src.ibm_cloud import get_wml_client, store_model_in_wml, deploy_model_in_wml

    wml = get_wml_client()
    if wml is None:
        logger.warning("IBM WML client not available – skipping deployment.")
        return

    model_uid = store_model_in_wml(pipeline, wml_client=wml)
    if model_uid:
        deploy_model_in_wml(model_uid, wml_client=wml)


def _run_inference_demo(pipeline) -> None:
    """Score demo samples and log predictions."""
    from src.predict import predict_batch

    logger.info("Scoring %d demo samples …", len(DEMO_SAMPLES))
    results = predict_batch(pipeline, DEMO_SAMPLES)
    logger.info("\n── Inference Results ───────────────────────────────────────")
    for i, row in results.iterrows():
        sample_type = DEMO_SAMPLES[i]["Type"]
        logger.info(
            "  Sample %d (Type=%s) → Predicted: %s",
            i + 1,
            sample_type,
            row["predicted_failure_type"],
        )


# ── Predict-only mode ─────────────────────────────────────────────────────────

def run_predict_only() -> None:
    """Load a saved model and score the demo samples."""
    logger.info("[PREDICT-ONLY] Loading saved model …")
    try:
        pipeline = load_model()
    except FileNotFoundError as exc:
        logger.error("%s – run `python main.py` first to train the model.", exc)
        sys.exit(1)

    _run_inference_demo(pipeline)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predictive Maintenance – Industrial Machinery (AICTE #39)"
    )
    parser.add_argument(
        "--skip-ibm",
        action="store_true",
        help="Skip IBM Cloud upload/deployment steps (local pipeline only).",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip training; load saved model and score demo samples.",
    )
    args = parser.parse_args()

    if args.predict_only:
        run_predict_only()
    else:
        run_pipeline(skip_ibm=args.skip_ibm)
