"""
src/model_training.py
Trains and compares multiple classifiers to predict industrial machine
failure type. Selects the best model via cross-validated F1-macro score.

Classifiers evaluated
─────────────────────
1. Random Forest
2. Gradient Boosted Trees (XGBoost)
3. LightGBM
4. Support Vector Machine (with calibration)
"""

import logging
import os
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from src.config import config
from src.preprocessing import build_preprocessor

logger = logging.getLogger(__name__)

os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)


# ── Candidate classifiers ─────────────────────────────────────────────────────

def _get_candidates(random_state: int) -> dict:
    """Return a dict of {name: classifier} candidates."""
    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBClassifier

        candidates["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )
    except ImportError:
        logger.warning("xgboost not installed – skipping XGBoost candidate.")

    try:
        from lightgbm import LGBMClassifier

        candidates["LightGBM"] = LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=63,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        logger.warning("lightgbm not installed – skipping LightGBM candidate.")

    svc_base = SVC(kernel="rbf", probability=False, class_weight="balanced", random_state=random_state)
    candidates["SVM"] = CalibratedClassifierCV(svc_base, cv=3)

    return candidates


# ── Pipeline builder ──────────────────────────────────────────────────────────

def build_pipeline(
    classifier,
    numeric_features: list,
    categorical_features: list,
) -> Pipeline:
    """Wrap preprocessor + classifier into a single sklearn Pipeline."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_select(
    X_train,
    y_train,
    numeric_features: list,
    categorical_features: list,
    cv_folds: int = 5,
    random_state: int = None,
) -> tuple[Pipeline, dict]:
    """
    Train all candidates using StratifiedKFold CV, select the best by
    macro F1-score, retrain it on the full training set.

    Returns
    -------
    best_pipeline : fitted sklearn Pipeline
    cv_results    : dict of {model_name: mean_cv_f1}
    """
    random_state = random_state or config.RANDOM_STATE
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    candidates = _get_candidates(random_state)

    cv_results = {}
    for name, clf in candidates.items():
        pipeline = build_pipeline(clf, numeric_features, categorical_features)
        logger.info("Cross-validating %s …", name)
        t0 = time.time()
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=skf,
            scoring="f1_macro",
            n_jobs=-1,
        )
        elapsed = time.time() - t0
        mean_f1 = scores.mean()
        cv_results[name] = round(float(mean_f1), 4)
        logger.info(
            "%s – CV F1-macro: %.4f ± %.4f  (%.1fs)",
            name,
            mean_f1,
            scores.std(),
            elapsed,
        )

    best_name = max(cv_results, key=cv_results.get)
    logger.info("Best model: %s (F1-macro: %.4f)", best_name, cv_results[best_name])

    best_clf = candidates[best_name]
    best_pipeline = build_pipeline(best_clf, numeric_features, categorical_features)
    logger.info("Retraining %s on full training set …", best_name)
    best_pipeline.fit(X_train, y_train)

    return best_pipeline, cv_results


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(pipeline: Pipeline, name: str = None) -> str:
    """Serialize the fitted pipeline to disk. Returns the file path."""
    name = name or config.MODEL_NAME
    path = os.path.join(config.MODEL_SAVE_PATH, f"{name}.pkl")
    joblib.dump(pipeline, path)
    logger.info("Model saved to: %s", path)
    return path


def load_model(name: str = None) -> Pipeline:
    """Deserialize a previously saved pipeline from disk."""
    name = name or config.MODEL_NAME
    path = os.path.join(config.MODEL_SAVE_PATH, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model found at '{path}'.")
    pipeline = joblib.load(path)
    logger.info("Model loaded from: %s", path)
    return pipeline
