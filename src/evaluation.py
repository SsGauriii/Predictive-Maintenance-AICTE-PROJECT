"""
src/evaluation.py
Model evaluation: classification report, confusion matrix,
per-class metrics, and feature importance extraction.
Saves all artefacts to the outputs/ directory.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.config import config
from src.preprocessing import LABEL_INVERSE_MAP

logger = logging.getLogger(__name__)

os.makedirs(config.OUTPUT_DIR, exist_ok=True)


# ── Core metrics ──────────────────────────────────────────────────────────────

def evaluate_model(pipeline, X_test, y_test) -> dict:
    """
    Compute and log a comprehensive set of classification metrics.

    Returns a dict with accuracy, macro/weighted F1, precision, recall,
    and per-class F1 scores.
    """
    y_pred = pipeline.predict(X_test)
    class_names = [LABEL_INVERSE_MAP.get(i, str(i)) for i in range(len(config.FAILURE_CLASSES))]
    labels_present = sorted(np.unique(np.concatenate([y_test, y_pred])))

    report_str = classification_report(
        y_test, y_pred,
        labels=labels_present,
        target_names=[class_names[i] for i in labels_present],
        zero_division=0,
    )
    logger.info("\nClassification Report:\n%s", report_str)

    report_dict = classification_report(
        y_test, y_pred,
        labels=labels_present,
        target_names=[class_names[i] for i in labels_present],
        zero_division=0,
        output_dict=True,
    )

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "precision_macro": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "per_class": report_dict,
    }

    # Multi-class ROC-AUC (OvR)
    if hasattr(pipeline, "predict_proba"):
        try:
            y_prob = pipeline.predict_proba(X_test)
            y_bin = label_binarize(y_test, classes=sorted(np.unique(y_test)))
            if y_bin.shape[1] > 1:
                roc_auc = roc_auc_score(y_bin, y_prob[:, :y_bin.shape[1]], multi_class="ovr", average="macro")
                metrics["roc_auc_macro"] = round(float(roc_auc), 4)
        except Exception as exc:
            logger.warning("Could not compute ROC-AUC: %s", exc)

    logger.info("Accuracy: %.4f | F1-macro: %.4f", metrics["accuracy"], metrics["f1_macro"])
    return metrics


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(pipeline, X_test, y_test, save: bool = True) -> str:
    """Plot and optionally save a normalised confusion matrix heatmap."""
    y_pred = pipeline.predict(X_test)
    class_names = [LABEL_INVERSE_MAP.get(i, str(i)) for i in range(len(config.FAILURE_CLASSES))]
    labels_present = sorted(np.unique(np.concatenate([y_test, y_pred])))
    present_names = [class_names[i] for i in labels_present]

    cm = confusion_matrix(y_test, y_pred, labels=labels_present, normalize="true")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=present_names,
        yticklabels=present_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Normalised Confusion Matrix – Failure Type Classifier", fontsize=14)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = ""
    if save:
        path = os.path.join(config.OUTPUT_DIR, "confusion_matrix.png")
        fig.savefig(path, dpi=150)
        logger.info("Confusion matrix saved to: %s", path)
    plt.close(fig)
    return path


# ── Feature importance ────────────────────────────────────────────────────────

def plot_feature_importance(pipeline, feature_names: list, save: bool = True) -> str:
    """
    Extract and plot feature importances from the best estimator.
    Supports RandomForest, XGBoost, LightGBM (tree-based feature_importances_).
    """
    classifier = pipeline.named_steps.get("classifier")
    base_clf = getattr(classifier, "estimator", classifier)  # unwrap calibrated SVC

    if not hasattr(base_clf, "feature_importances_"):
        logger.warning("Classifier does not expose feature_importances_ – skipping plot.")
        return ""

    importances = base_clf.feature_importances_
    n_feats = len(importances)

    # Align feature names (preprocessor may expand feature count)
    if len(feature_names) != n_feats:
        feature_names = [f"feature_{i}" for i in range(n_feats)]

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=True)
        .tail(20)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(importance_df["feature"], importance_df["importance"], color="#3b82d4")
    ax.set_title("Top Feature Importances – Predictive Maintenance", fontsize=13)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    plt.tight_layout()

    path = ""
    if save:
        path = os.path.join(config.OUTPUT_DIR, "feature_importance.png")
        fig.savefig(path, dpi=150)
        logger.info("Feature importance plot saved to: %s", path)
    plt.close(fig)
    return path


# ── CV results bar chart ──────────────────────────────────────────────────────

def plot_cv_results(cv_results: dict, save: bool = True) -> str:
    """Bar chart comparing cross-validated F1-macro scores of all candidates."""
    names = list(cv_results.keys())
    scores = [cv_results[n] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, scores, color=["#3b82d4", "#7c5cd8", "#e5a020", "#e53a3a"][: len(names)])
    ax.set_ylim(0, 1.0)
    ax.set_title("Model Comparison – CV F1-macro Score", fontsize=13)
    ax.set_ylabel("F1-macro")
    ax.set_xlabel("Classifier")
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()

    path = ""
    if save:
        path = os.path.join(config.OUTPUT_DIR, "cv_comparison.png")
        fig.savefig(path, dpi=150)
        logger.info("CV comparison chart saved to: %s", path)
    plt.close(fig)
    return path


# ── Class distribution ────────────────────────────────────────────────────────

def plot_class_distribution(y: np.ndarray, title: str = "Class Distribution", save: bool = True) -> str:
    """Bar chart showing samples per failure class."""
    class_names = [LABEL_INVERSE_MAP.get(i, str(i)) for i in range(len(config.FAILURE_CLASSES))]
    unique, counts = np.unique(y, return_counts=True)
    labels = [class_names[i] for i in unique]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, counts, color="#3b82d4")
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Sample Count")
    ax.set_xlabel("Failure Type")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    path = ""
    if save:
        fname = title.lower().replace(" ", "_") + ".png"
        path = os.path.join(config.OUTPUT_DIR, fname)
        fig.savefig(path, dpi=150)
        logger.info("Class distribution chart saved to: %s", path)
    plt.close(fig)
    return path
