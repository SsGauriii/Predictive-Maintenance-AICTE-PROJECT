"""
src/config.py
Centralized configuration loader from .env / environment variables.
IBM Cloud Lite Services – connection settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── IBM Watson Machine Learning ────────────────────────────────────────
    WML_API_KEY: str = os.getenv("IBM_WML_API_KEY", "")
    WML_URL: str = os.getenv("IBM_WML_URL", "https://us-south.ml.cloud.ibm.com")
    WML_INSTANCE_ID: str = os.getenv("IBM_WML_INSTANCE_ID", "")
    WML_SPACE_ID: str = os.getenv("IBM_WML_SPACE_ID", "")

    # ── IBM Cloud Object Storage ───────────────────────────────────────────
    COS_API_KEY: str = os.getenv("IBM_COS_API_KEY", "")
    COS_SERVICE_INSTANCE_ID: str = os.getenv("IBM_COS_SERVICE_INSTANCE_ID", "")
    COS_ENDPOINT: str = os.getenv(
        "IBM_COS_ENDPOINT",
        "https://s3.us-south.cloud-object-storage.appdomain.cloud",
    )
    COS_AUTH_ENDPOINT: str = os.getenv(
        "IBM_COS_AUTH_ENDPOINT", "https://iam.cloud.ibm.com/oidc/token"
    )
    COS_BUCKET: str = os.getenv("IBM_COS_BUCKET_NAME", "predictive-maintenance-bucket")

    # ── Watson Studio ──────────────────────────────────────────────────────
    WATSON_PROJECT_ID: str = os.getenv("IBM_WATSON_STUDIO_PROJECT_ID", "")

    # ── Dataset ────────────────────────────────────────────────────────────
    DATASET_PATH: str = os.getenv("DATASET_PATH", "data/predictive_maintenance.csv")

    # ── Model ──────────────────────────────────────────────────────────────
    MODEL_NAME: str = os.getenv("MODEL_NAME", "predictive_maintenance_classifier")
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "1.0")
    RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))
    TEST_SIZE: float = float(os.getenv("TEST_SIZE", "0.2"))

    # ── App ────────────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "outputs")
    MODEL_SAVE_PATH: str = os.getenv("MODEL_SAVE_PATH", "models/")

    # ── Target labels ──────────────────────────────────────────────────────
    TARGET_COLUMN: str = "Failure Type"
    FAILURE_CLASSES: list = [
        "No Failure",
        "Heat Dissipation Failure",
        "Power Failure",
        "Overstrain Failure",
        "Tool Wear Failure",
        "Random Failures",
    ]

    # ── Feature columns ────────────────────────────────────────────────────
    NUMERIC_FEATURES: list = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    CATEGORICAL_FEATURES: list = ["Type"]


config = Config()
