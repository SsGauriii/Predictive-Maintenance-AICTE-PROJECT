"""
src/ibm_cloud.py
IBM Cloud Lite Services integration layer.

Services used (all on the Free / Lite tier)
────────────────────────────────────────────
1. IBM Watson Machine Learning  – deploy & score the trained model
2. IBM Cloud Object Storage     – store datasets and model artefacts

Credentials are read from environment variables (see .env / env.example).
"""

import io
import logging
import os

from src.config import config

logger = logging.getLogger(__name__)


# ── IBM Cloud Object Storage ──────────────────────────────────────────────────

def get_cos_client():
    """
    Create and return an IBM Cloud Object Storage (ibm_boto3) client.

    Requires environment variables:
        IBM_COS_API_KEY, IBM_COS_SERVICE_INSTANCE_ID,
        IBM_COS_ENDPOINT, IBM_COS_AUTH_ENDPOINT
    """
    try:
        import ibm_boto3
        from ibm_botocore.client import Config as BotoConfig
    except ImportError:
        logger.error(
            "ibm-cos-sdk is not installed. Run: pip install ibm-cos-sdk"
        )
        return None

    if not config.COS_API_KEY:
        logger.warning(
            "IBM_COS_API_KEY is not set – COS client unavailable."
        )
        return None

    client = ibm_boto3.client(
        "s3",
        ibm_api_key_id=config.COS_API_KEY,
        ibm_service_instance_id=config.COS_SERVICE_INSTANCE_ID,
        ibm_auth_endpoint=config.COS_AUTH_ENDPOINT,
        config=BotoConfig(signature_version="oauth"),
        endpoint_url=config.COS_ENDPOINT,
    )
    logger.info("IBM COS client initialised (endpoint: %s).", config.COS_ENDPOINT)
    return client


def ensure_bucket_exists(cos_client) -> bool:
    """Create the COS bucket if it does not already exist."""
    try:
        cos_client.head_bucket(Bucket=config.COS_BUCKET)
        logger.info("COS bucket '%s' already exists.", config.COS_BUCKET)
        return True
    except Exception:
        pass
    try:
        cos_client.create_bucket(Bucket=config.COS_BUCKET)
        logger.info("COS bucket '%s' created.", config.COS_BUCKET)
        return True
    except Exception as exc:
        logger.error("Could not create COS bucket '%s': %s", config.COS_BUCKET, exc)
        return False


def upload_file_to_cos(local_path: str, object_key: str = None, cos_client=None) -> bool:
    """
    Upload a local file to the configured COS bucket.
    Auto-creates the bucket if it does not exist.

    Parameters
    ----------
    local_path  : Path to the local file.
    object_key  : Target key in the bucket (defaults to the file basename).
    cos_client  : Optionally pass an existing client; one is created if None.

    Returns True on success, False on failure.
    """
    cos_client = cos_client or get_cos_client()
    if cos_client is None:
        return False

    ensure_bucket_exists(cos_client)
    object_key = object_key or os.path.basename(local_path)

    try:
        with open(local_path, "rb") as fh:
            cos_client.upload_fileobj(fh, config.COS_BUCKET, object_key)
        logger.info("Uploaded '%s' to COS bucket '%s' as '%s'.", local_path, config.COS_BUCKET, object_key)
        return True
    except Exception as exc:
        logger.error("COS upload failed: %s", exc)
        return False


def download_file_from_cos(object_key: str, local_path: str, cos_client=None) -> bool:
    """
    Download an object from COS to a local file.

    Returns True on success, False on failure.
    """
    cos_client = cos_client or get_cos_client()
    if cos_client is None:
        return False

    try:
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        cos_client.download_file(config.COS_BUCKET, object_key, local_path)
        logger.info("Downloaded '%s' from COS to '%s'.", object_key, local_path)
        return True
    except Exception as exc:
        logger.error("COS download failed: %s", exc)
        return False


# ── IBM Watson Machine Learning ───────────────────────────────────────────────

def get_wml_client():
    """
    Create and return an IBM Watson Machine Learning (APIClient) client.

    Requires environment variables:
        IBM_WML_API_KEY, IBM_WML_URL, IBM_WML_SPACE_ID
    """
    try:
        from ibm_watsonx_ai import APIClient
    except ImportError:
        logger.error(
            "ibm-watsonx-ai is not installed. "
            "Run: pip install ibm-watsonx-ai"
        )
        return None

    if not config.WML_API_KEY:
        logger.warning("IBM_WML_API_KEY is not set – WML client unavailable.")
        return None

    wml_credentials = {
        "url": config.WML_URL,
        "apikey": config.WML_API_KEY,
    }
    client = APIClient(wml_credentials)

    if config.WML_SPACE_ID:
        client.set.default_space(config.WML_SPACE_ID)
        logger.info("WML client initialised – deployment space: %s.", config.WML_SPACE_ID)
    else:
        logger.warning("IBM_WML_SPACE_ID not set. Set a deployment space before deploying.")

    return client


def store_model_in_wml(pipeline, model_name: str = None, wml_client=None) -> str:
    """
    Store a trained scikit-learn pipeline in IBM Watson Machine Learning.

    NOTE (IBM WML Lite – 2026):
    IBM retired ALL framework-specific software specs (scikit-learn, xgboost,
    pytorch-onnx, tensorflow) for Lite plan accounts. Only runtime-25.1-py3.12
    remains active but it does not accept any model type mapping via the v4 API.
    Manual model uploads via Watson Studio UI also only accept .csv data files.

    For AICTE project demonstration purposes, this function logs the limitation
    and returns an empty string. The model is already stored in IBM COS (Step 8)
    and can be used for local inference (Step 10).

    To deploy on WML: upgrade to a paid plan or use IBM AutoAI in Watson Studio
    to train and deploy a model directly from the CSV dataset stored in COS.

    Returns empty string (IBM WML Lite model storage not available).
    """
    model_name = model_name or config.MODEL_NAME
    logger.warning(
        "IBM WML Lite (2026): All sklearn/xgboost/pytorch-onnx software specs "
        "have been retired. Model '%s' cannot be stored via the WML v4 API on "
        "a Lite plan. The model is saved locally at '%s' and uploaded to IBM COS. "
        "To deploy on WML, use a paid plan or IBM AutoAI in Watson Studio.",
        model_name,
        os.path.join(config.MODEL_SAVE_PATH, f"{model_name}.pkl"),
    )
    return ""


def _store_model_in_wml_paid(pipeline, model_name: str = None, wml_client=None) -> str:
    """
    (Paid plan only) Store a trained scikit-learn pipeline in IBM WML.
    Kept here for reference — requires a Standard/Professional WML plan
    where scikit-learn_1.x software specs are available.
    """
    import tarfile, tempfile, shutil

    wml_client = wml_client or get_wml_client()
    if wml_client is None:
        logger.warning("WML client unavailable – model not stored in WML.")
        return ""

    model_name = model_name or config.MODEL_NAME

    # ── Convert pipeline to ONNX ──────────────────────────────────────────
    tmp_dir = tempfile.mkdtemp()
    onnx_path = os.path.join(tmp_dir, "model.onnx")
    tar_path  = os.path.join(tmp_dir, "model.tar.gz")

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType, StringTensorType

        # Build initial_types matching the pipeline's feature schema
        # 9 numeric features + 4 engineered = 13 numeric, 1 categorical
        n_numeric = len(config.NUMERIC_FEATURES) + 4  # +4 engineered features
        initial_types = [
            ("input_numeric", FloatTensorType([None, n_numeric])),
            ("input_type",    StringTensorType([None, 1])),
        ]
        onnx_model = convert_sklearn(
            pipeline,
            initial_types=initial_types,
        )
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        logger.info("Pipeline converted to ONNX: %s", onnx_path)

    except Exception as exc:
        logger.warning("ONNX conversion failed (%s). Falling back to pkl tar.gz.", exc)
        import joblib
        pkl_path = os.path.join(tmp_dir, "model.pkl")
        joblib.dump(pipeline, pkl_path)
        onnx_path = pkl_path  # reuse path variable

    # ── Package as tar.gz ─────────────────────────────────────────────────
    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(onnx_path, arcname=os.path.basename(onnx_path))
        logger.info("Model packaged: %s", tar_path)
    except Exception as exc:
        logger.error("Packaging failed: %s", exc)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return ""

    # ── Upload to WML ─────────────────────────────────────────────────────
    mn = wml_client._models.ConfigurationMetaNames

    # pytorch-onnx_2.1 works with runtime-25.1-py3.12 (confirmed by IBM docs)
    spec_uid = wml_client.software_specifications.get_uid_by_name("runtime-25.1-py3.12")
    model_type = "pytorch-onnx_2.1"

    model_props = {
        mn.NAME: model_name,
        mn.TYPE: model_type,
        mn.SOFTWARE_SPEC_ID: spec_uid,
    }

    try:
        stored_model = wml_client._models.store(model=tar_path, meta_props=model_props)
        model_uid = wml_client._models.get_id(stored_model)
        logger.info("Model stored in WML – UID: %s", model_uid)
        return model_uid
    except Exception as exc:
        logger.error("Failed to store model in WML: %s", exc)
        return ""
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def deploy_model_in_wml(model_uid: str, deployment_name: str = None, wml_client=None) -> str:
    """
    Create an online deployment for a stored WML model.

    Returns the deployment UID, or an empty string on failure.
    """
    wml_client = wml_client or get_wml_client()
    if wml_client is None or not model_uid:
        return ""

    deployment_name = deployment_name or f"{config.MODEL_NAME}_deployment"

    deploy_props = {
        wml_client.deployments.ConfigurationMetaNames.NAME: deployment_name,
        wml_client.deployments.ConfigurationMetaNames.ONLINE: {},
        wml_client.deployments.ConfigurationMetaNames.SERVING_NAME: deployment_name.lower().replace(" ", "_"),
    }

    try:
        deployment = wml_client.deployments.create(model_uid, meta_props=deploy_props)
        deployment_uid = wml_client.deployments.get_uid(deployment)
        logger.info("Model deployed in WML – deployment UID: %s", deployment_uid)
        return deployment_uid
    except Exception as exc:
        logger.error("Failed to deploy model in WML: %s", exc)
        return ""


def score_via_wml(deployment_uid: str, input_data: list, wml_client=None) -> list:
    """
    Send a scoring request to an IBM WML online deployment.

    Parameters
    ----------
    deployment_uid : UID of the deployed model.
    input_data     : List of feature value lists (one row per prediction).
    wml_client     : Optional existing client.

    Returns a list of predicted class labels (integers).
    """
    wml_client = wml_client or get_wml_client()
    if wml_client is None or not deployment_uid:
        return []

    scoring_payload = {
        "input_data": [{"values": input_data}]
    }

    try:
        predictions = wml_client.deployments.score(deployment_uid, scoring_payload)
        results = [row[0] for row in predictions["predictions"][0]["values"]]
        logger.info("WML scoring complete – %d predictions returned.", len(results))
        return results
    except Exception as exc:
        logger.error("WML scoring request failed: %s", exc)
        return []
