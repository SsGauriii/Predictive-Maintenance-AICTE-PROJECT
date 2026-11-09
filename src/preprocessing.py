"""
src/preprocessing.py
Data cleaning, class-balance handling (SMOTE), train/test split,
and sklearn Pipeline construction for the feature engineering steps.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from src.config import config

logger = logging.getLogger(__name__)


# ── Label encoding map ────────────────────────────────────────────────────────

FAILURE_LABEL_MAP = {label: idx for idx, label in enumerate(config.FAILURE_CLASSES)}
LABEL_INVERSE_MAP = {v: k for k, v in FAILURE_LABEL_MAP.items()}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop identifier columns, handle missing values, and validate schema.

    Returns a cleaned DataFrame ready for feature engineering.
    """
    df = df.copy()

    # Drop non-predictive identifiers
    drop_cols = [c for c in ["UDI", "Product ID"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # Drop individual failure flag columns – target is 'Failure Type'
    flag_cols = [c for c in ["TWF", "HDF", "PWF", "OSF", "RNF", "Machine failure"] if c in df.columns]
    df.drop(columns=flag_cols, inplace=True)

    # Verify required columns exist
    required = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES + [config.TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Drop rows with nulls in required columns
    before = len(df)
    df.dropna(subset=required, inplace=True)
    after = len(df)
    if before != after:
        logger.warning("Dropped %d rows with null values.", before - after)

    logger.info("Cleaned dataset shape: %s", df.shape)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive domain-driven engineered features from raw sensor readings.

    New features
    ────────────
    power_W            – mechanical power in watts (ω × T)
    temp_delta_K       – process minus air temperature difference
    tool_torque_product– tool wear × torque (overstrain proxy)
    speed_torque_ratio – rotational speed / torque
    """
    df = df.copy()

    omega = df["Rotational speed [rpm]"] * (2 * np.pi / 60)
    df["power_W"] = (omega * df["Torque [Nm]"]).round(2)
    df["temp_delta_K"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    ).round(2)
    df["tool_torque_product"] = (
        df["Tool wear [min]"] * df["Torque [Nm]"]
    ).round(2)
    df["speed_torque_ratio"] = (
        df["Rotational speed [rpm]"] / df["Torque [Nm]"].replace(0, np.nan)
    ).round(4)

    logger.info("Engineered features added. New shape: %s", df.shape)
    return df


def encode_target(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Encode the target column to integer labels.

    Returns (df_without_target, y_encoded).
    """
    y_raw = df[config.TARGET_COLUMN].copy()
    y = y_raw.map(FAILURE_LABEL_MAP)

    # Handle unseen labels gracefully
    unseen = y_raw[y.isna()].unique().tolist()
    if unseen:
        logger.warning("Unknown failure types found and set to 0: %s", unseen)
        y.fillna(0, inplace=True)

    X = df.drop(columns=[config.TARGET_COLUMN])
    return X, y.astype(int).values


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - Standardizes numeric features.
      - Ordinal-encodes the machine type ('L', 'M', 'H').
    """
    numeric_pipeline = Pipeline([("scaler", StandardScaler())])

    categorical_pipeline = Pipeline(
        [
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[["L", "M", "H"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="passthrough",
    )
    return preprocessor


def split_data(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = None,
    random_state: int = None,
) -> tuple:
    """
    Stratified train / test split.

    Returns (X_train, X_test, y_train, y_test).
    """
    test_size = test_size or config.TEST_SIZE
    random_state = random_state or config.RANDOM_STATE

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        "Train size: %d | Test size: %d", len(X_train), len(X_test)
    )
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state: int = None):
    """
    Apply SMOTE to handle severe class imbalance in the training set.

    Returns (X_resampled, y_resampled).
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        logger.warning("imbalanced-learn not installed – skipping SMOTE.")
        return X_train, y_train

    random_state = random_state or config.RANDOM_STATE
    sm = SMOTE(random_state=random_state, k_neighbors=3)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    logger.info(
        "SMOTE applied: %d -> %d training samples.", len(y_train), len(y_res)
    )
    return X_res, y_res
