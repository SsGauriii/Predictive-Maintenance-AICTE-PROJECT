"""
src/data_loader.py
Loads the AI4I 2020 Predictive Maintenance dataset.
Supports local CSV file or download from a URL.
Optionally uploads raw data to IBM Cloud Object Storage.
"""

import logging
import os

import pandas as pd
import requests

from src.config import config

logger = logging.getLogger(__name__)


# ── Column definitions ────────────────────────────────────────────────────────

COLUMN_RENAME_MAP = {
    "UDI": "UDI",
    "Product ID": "Product ID",
    "Type": "Type",
    "Air temperature [K]": "Air temperature [K]",
    "Process temperature [K]": "Process temperature [K]",
    "Rotational speed [rpm]": "Rotational speed [rpm]",
    "Torque [Nm]": "Torque [Nm]",
    "Tool wear [min]": "Tool wear [min]",
    "Machine failure": "Machine failure",
    "TWF": "TWF",
    "HDF": "HDF",
    "PWF": "PWF",
    "OSF": "OSF",
    "RNF": "RNF",
    "Failure Type": "Failure Type",
}


def load_dataset(path: str = None) -> pd.DataFrame:
    """
    Load the predictive maintenance dataset from a local CSV file.
    If the file does not exist, creates a synthetic representative sample
    so the pipeline can be tested end-to-end without the Kaggle download.

    Returns
    -------
    pd.DataFrame – raw dataset with all original columns.
    """
    path = path or config.DATASET_PATH

    if os.path.exists(path):
        logger.info("Loading dataset from local path: %s", path)
        df = pd.read_csv(path)
        logger.info("Dataset loaded – shape: %s", df.shape)
        return df

    logger.warning(
        "Dataset not found at '%s'. Generating synthetic data for demonstration.", path
    )
    return _generate_synthetic_data()


def _generate_synthetic_data(n_samples: int = 10_000) -> pd.DataFrame:
    """
    Generate a synthetic dataset that mirrors the AI4I 2020 schema.
    Column distributions are approximate – intended for pipeline validation only.
    """
    import numpy as np

    rng = np.random.default_rng(config.RANDOM_STATE)

    product_types = rng.choice(["L", "M", "H"], size=n_samples, p=[0.50, 0.30, 0.20])

    air_temp = rng.normal(300, 2, n_samples)
    proc_temp = air_temp + rng.normal(10, 1, n_samples)
    rot_speed = rng.normal(1500, 200, n_samples).clip(1168, 2886)
    torque = rng.normal(40, 10, n_samples).clip(3.8, 76.6)
    tool_wear = rng.integers(0, 253, n_samples)

    # Compute failure conditions matching the dataset description
    power = rot_speed * torque * (2 * 3.14159 / 60)
    temp_diff = proc_temp - air_temp

    twf = ((tool_wear >= 200) & (tool_wear <= 240)).astype(int)
    hdf = ((temp_diff < 8.6) & (rot_speed < 1380)).astype(int)
    pwf = ((power < 3500) | (power > 9000)).astype(int)
    osf_threshold = {t: v for t, v in zip(["L", "M", "H"], [11_000, 12_000, 13_000])}
    osf = (
        tool_wear * torque
        > pd.Series(product_types).map(osf_threshold).values
    ).astype(int)
    rnf = (rng.random(n_samples) < 0.001).astype(int)

    machine_failure = ((twf + hdf + pwf + osf + rnf) > 0).astype(int)

    # Assign failure type label (priority order)
    failure_type = pd.array(["No Failure"] * n_samples, dtype=object)
    failure_type[rnf == 1] = "Random Failures"
    failure_type[osf == 1] = "Overstrain Failure"
    failure_type[pwf == 1] = "Power Failure"
    failure_type[hdf == 1] = "Heat Dissipation Failure"
    failure_type[twf == 1] = "Tool Wear Failure"

    df = pd.DataFrame(
        {
            "UDI": range(1, n_samples + 1),
            "Product ID": [
                f"{t}{rng.integers(10000, 99999)}"
                for t in product_types
            ],
            "Type": product_types,
            "Air temperature [K]": air_temp.round(1),
            "Process temperature [K]": proc_temp.round(1),
            "Rotational speed [rpm]": rot_speed.astype(int),
            "Torque [Nm]": torque.round(1),
            "Tool wear [min]": tool_wear,
            "Machine failure": machine_failure,
            "TWF": twf,
            "HDF": hdf,
            "PWF": pwf,
            "OSF": osf,
            "RNF": rnf,
            "Failure Type": failure_type,
        }
    )
    logger.info("Synthetic dataset generated – shape: %s", df.shape)
    return df


def upload_to_cos(df: pd.DataFrame, cos_client, filename: str = "raw_data.csv") -> None:
    """
    Upload the raw dataset as a CSV to IBM Cloud Object Storage.

    Parameters
    ----------
    df         : DataFrame to upload.
    cos_client : Initialised ibm_boto3 client (from ibm_cloud.py).
    filename   : Target object key in the COS bucket.
    """
    import io

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    cos_client.put_object(
        Bucket=config.COS_BUCKET,
        Key=filename,
        Body=csv_bytes,
    )
    logger.info(
        "Uploaded '%s' (%d rows) to COS bucket '%s'.",
        filename,
        len(df),
        config.COS_BUCKET,
    )
