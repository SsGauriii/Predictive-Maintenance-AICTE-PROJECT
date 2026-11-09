"""
src/predict.py
Inference utilities – single-sample and batch prediction using either
a local saved model or an IBM WML online deployment.
"""

import logging

import numpy as np
import pandas as pd

from src.config import config
from src.preprocessing import LABEL_INVERSE_MAP

logger = logging.getLogger(__name__)


def predict_local(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """
    Run inference using the local sklearn Pipeline.

    Parameters
    ----------
    pipeline : Fitted sklearn Pipeline.
    X        : DataFrame with the same feature columns used during training.

    Returns a DataFrame with columns:
        predicted_label (int), predicted_failure_type (str),
        and one probability column per failure class.
    """
    labels = pipeline.predict(X)
    failure_types = [LABEL_INVERSE_MAP.get(int(l), "Unknown") for l in labels]

    result = pd.DataFrame(
        {
            "predicted_label": labels,
            "predicted_failure_type": failure_types,
        }
    )

    if hasattr(pipeline, "predict_proba"):
        try:
            probs = pipeline.predict_proba(X)
            for idx, cls_name in enumerate(config.FAILURE_CLASSES):
                col = f"prob_{cls_name.lower().replace(' ', '_')}"
                if idx < probs.shape[1]:
                    result[col] = probs[:, idx].round(4)
        except Exception as exc:
            logger.warning("Could not compute class probabilities: %s", exc)

    return result


def predict_single(pipeline, sensor_data: dict) -> dict:
    """
    Predict the failure type for a single machine's sensor readings.

    Parameters
    ----------
    pipeline    : Fitted sklearn Pipeline.
    sensor_data : Dict with keys matching feature column names, e.g.:
                  {
                      "Type": "M",
                      "Air temperature [K]": 298.1,
                      "Process temperature [K]": 308.6,
                      "Rotational speed [rpm]": 1551,
                      "Torque [Nm]": 42.8,
                      "Tool wear [min]": 0
                  }

    Returns a dict with 'predicted_failure_type' and class probabilities.
    """
    # Ensure engineered features are derived if missing
    _add_derived_features(sensor_data)

    X = pd.DataFrame([sensor_data])
    result_df = predict_local(pipeline, X)
    return result_df.iloc[0].to_dict()


def predict_batch(pipeline, records: list[dict]) -> pd.DataFrame:
    """
    Predict failure types for a batch of sensor records.

    Parameters
    ----------
    records : List of dicts, each matching sensor_data format.

    Returns a DataFrame with prediction results appended.
    """
    for record in records:
        _add_derived_features(record)

    X = pd.DataFrame(records)
    return predict_local(pipeline, X)


def _add_derived_features(sensor_data: dict) -> None:
    """
    Derive engineered features in-place if they are absent.
    Mirrors the logic in preprocessing.engineer_features().
    """
    rpm = sensor_data.get("Rotational speed [rpm]", 0)
    torque = sensor_data.get("Torque [Nm]", 1)
    air_temp = sensor_data.get("Air temperature [K]", 300)
    proc_temp = sensor_data.get("Process temperature [K]", 310)
    tool_wear = sensor_data.get("Tool wear [min]", 0)

    omega = rpm * (2 * 3.14159265 / 60)
    sensor_data.setdefault("power_W", round(omega * torque, 2))
    sensor_data.setdefault("temp_delta_K", round(proc_temp - air_temp, 2))
    sensor_data.setdefault("tool_torque_product", round(tool_wear * torque, 2))
    sensor_data.setdefault(
        "speed_torque_ratio", round(rpm / torque if torque != 0 else 0, 4)
    )
