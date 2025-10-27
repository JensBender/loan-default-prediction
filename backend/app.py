# --- Imports ---
# Standard library imports
from http import client
from pathlib import Path
from typing import Any
import logging
import logging.config
import json
import uuid
import time
from datetime import datetime, timezone

# Third-party library imports
from fastapi import FastAPI, HTTPException, Request
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
import geoip2.database
from geoip2.errors import AddressNotFoundError

# Local imports
from backend.schemas import (
    PipelineInput,
    PredictionEnum,
    PredictedProbabilities,
    PredictionResult,
    PredictionResponse    
)
from src.custom_transformers import (
    MissingValueChecker, 
    MissingValueStandardizer, 
    RobustSimpleImputer,
    SnakeCaseFormatter, 
    BooleanColumnTransformer, 
    JobStabilityTransformer, 
    CityTierTransformer, 
    StateDefaultRateTargetEncoder,
    RobustStandardScaler,
    RobustOneHotEncoder,
    RobustOrdinalEncoder,
    FeatureSelector
)
from src.utils import get_root_directory

# --- Logging ---
# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Define logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "monitoring": {
            "format": "%(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",  # write to Python standard output stream, which goes to Docker container's standard output, which goes to Hugging Face host server, which goes to Hugging Face Space Logs tab
        },
        "monitoring_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "monitoring",
            "filename": str(log_dir / "prediction_logs.jsonl"),  # JSON Lines format: one JSON object per line
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 3,  # will create prediction_logs.jsonl.1, .2, and .3 for max 4 log files (40 MB), then overwrite
        },
    },
    "loggers": {
        "": {  # root logger for general logs
            "handlers": ["console"],
            "level": "INFO",
        },
        "monitoring": {  # prediction records logger for model monitoring 
            "handlers": ["monitoring_file"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

# Apply logging configuration
logging.config.dictConfig(LOGGING_CONFIG)

# Get loggers
logger = logging.getLogger(__name__)
monitoring_logger = logging.getLogger("monitoring")


# --- Helper Functions ---
# Function to get batch-level metadata for logging
def get_batch_metadata(
    pipeline_input_dict_ls: list[dict[str, Any]], 
    request: Request, 
    geoip_reader: geoip2.database.Reader | None
) -> dict[str, Any]:    
    # Get user agent and IP from frontend
    user_agent = pipeline_input_dict_ls[0].get("user_agent", None)  # use first input in list of inputs
    if user_agent is None:  # fall back for direct API request to backend
        user_agent = request.headers.get("user-agent", "unknown")  # get from request headers of FastAPI backend
    client_ip = pipeline_input_dict_ls[0].get("client_ip", None)  
    if client_ip is None:
        x_forwarded_for = request.headers.get("x-forwarded-for")  # single str with one or more comma-separated IP addresses
        client_ip = x_forwarded_for.split(",")[0].strip() if x_forwarded_for else request.client.host  # first IP is client IP address
    
    # Get client country from IP address
    client_country = "unknown"
    if geoip_reader and client_ip:
        try:
            response = geoip_reader.country(client_ip)
            client_country = response.country.name
        except AddressNotFoundError:  # this occurs for unknown or private or reserved IPs (e.g., 127.0.0.1)
            logger.debug("IP address not found in GeoLite2 country database. Likely a private or local address.")

    # Create dictionary with batch-level metadata
    metadata = {
        "batch_id": str(uuid.uuid4()),
        "batch_size": len(pipeline_input_dict_ls),
        "batch_timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION_TAG,
        "client_country": client_country,
        "user_agent": user_agent,
    } 

    return metadata


# Function to load a scikit-learn pipeline from the local machine
def load_pipeline_from_local(path: str | Path) -> Pipeline:
    # Input type validation
    if not isinstance(path, (str, Path)):
        raise TypeError(f"Error when loading pipeline: 'path' must be a string or Path object, got {type(path).__name__}")

    # Get path as both string and Path object
    if isinstance(path, Path):
        path_str = str(path)
    else:  # isinstance(path, str)
        path_str = path 
        path = Path(path)

    # Ensure file exists
    if not path.exists():
        raise FileNotFoundError(f"Error when loading pipeline: File not found at '{path_str}'")
    
    # Load pipeline 
    try:
        logger.info(f"Loading pipeline from '{path_str}'...")
        pipeline = joblib.load(path_str)
        logger.info("Pipeline loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Error when loading pipeline from '{path_str}'") from e
    
    # Ensure loaded object is a scikit-learn Pipeline
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Error when loading pipeline: Loaded object is not a scikit-learn Pipeline")

    # Ensure pipeline has .predict_proba() method
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError("Error when loading pipeline: Loaded pipeline does not have a .predict_proba() method")

    return pipeline


# Function to download and load a scikit-learn pipeline from a Hugging Face Hub repository
def load_pipeline_from_huggingface(repo_id: str, filename: str, revision: str) -> Pipeline:
    try:
        # .hf_hub_download() downloads the pipeline file and returns its local file path (inside the Docker container)
        # if the pipeline file was already downloaded, it will use the cached pipeline that is already stored inside the Docker container 
        logger.info(
            f"Connecting to Hugging Face Hub to download pipeline file '{filename}' with tag '{revision}' in repo '{repo_id}'. "
            "If already cached, will use local copy."
        )
        pipeline_path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)

        # Load pipeline from file inside the Docker container
        pipeline = load_pipeline_from_local(pipeline_path)

        return pipeline

    except Exception as e:
        raise RuntimeError(f"Error loading pipeline '{filename}' from Hugging Face Hub repository '{repo_id}': {e}") from e 


# --- Geolocation Database ---
# Load the GeoLite2 country database to log client country for model monitoring (download database from https://www.maxmind.com to the "geoip_db/" directory)
GEO_DB_PATH = Path("geoip_db/GeoLite2-Country.mmdb")
try:
    geoip_reader = geoip2.database.Reader(GEO_DB_PATH)
    logger.info(f"Successfully loaded GeoLite2 country database from '{GEO_DB_PATH}'")
except FileNotFoundError:
    logger.error(f"GeoLite2 country database not found at '{GEO_DB_PATH}'. Client country will not be logged. Download the database from https://www.maxmind.com.")
    geoip_reader = None

# --- ML Pipeline ---
# Load loan default prediction pipeline (including data preprocessing and Random Forest Classifier model) from Hugging Face Hub
PIPELINE_VERSION_TAG = "v1.0"
pipeline = load_pipeline_from_huggingface(
    repo_id="JensBender/loan-default-prediction-pipeline", 
    filename="loan_default_rf_pipeline.joblib",
    revision=PIPELINE_VERSION_TAG  
)

# Load pipeline from local machine (use for local setup without Hugging Face Hub)
# root_dir = get_root_directory()  # get path to root directory
# pipeline_path = root_dir / "models" / "loan_default_rf_pipeline.joblib"  # get path to pipeline file
# pipeline = load_pipeline_from_local(pipeline_path)

# --- API ---
# Create FastAPI app
app = FastAPI()


# Prediction endpoint 
@app.post("/predict", response_model=PredictionResponse)
def predict(pipeline_input: PipelineInput | list[PipelineInput], request: Request) -> PredictionResponse:  # JSON object -> PipelineInput | JSON array -> list[PipelineInput]
    try:
        # Standardize input
        pipeline_input_dict_ls: list[dict[str, Any]]
        if isinstance(pipeline_input, list):
            if pipeline_input == []:  # handle empty batch input
                return PredictionResponse(results=[])
            pipeline_input_dict_ls = [input.model_dump() for input in pipeline_input]
        else:  # isinstance(pipeline_input, PipelineInput)
            pipeline_input_dict_ls = [pipeline_input.model_dump()]
        
        # Get batch metadata for logging
        batch_metadata = get_batch_metadata(pipeline_input_dict_ls, request, geoip_reader)

        # Remove "client_ip" and "user_agent" from pipeline input
        pipeline_input_cleaned = [
            {k: v for k, v in d.items() if k not in {"client_ip", "user_agent"}} 
            for d in pipeline_input_dict_ls
        ]

        # Create DataFrame
        pipeline_input_df: pd.DataFrame = pd.DataFrame(pipeline_input_cleaned)

        # Use pipeline to batch predict probabilities (and measure latency)
        start_time = time.perf_counter()  # use .perf_counter() for latency measurement and .time() for timestamps
        predicted_probabilities: np.ndarray = pipeline.predict_proba(pipeline_input_df)  
        pipeline_prediction_latency_ms = round((time.perf_counter() - start_time) * 1000)  # rounded to milliseconds

        # Apply optimized threshold to convert probabilities to binary predictions
        optimized_threshold: float = 0.29  # see threshold optimization in training script "loan_default_prediction.ipynb"
        predictions: np.ndarray = (predicted_probabilities[:, 1] >= optimized_threshold)  # bool 1d-array based on class 1 "Default"

        # Add latency to batch metadata for logging
        batch_metadata.update({
            "batch_latency_ms": pipeline_prediction_latency_ms,
            "avg_prediction_latency_ms": round(pipeline_prediction_latency_ms / len(pipeline_input_dict_ls)),
        })

        # --- Create prediction response --- 
        results: list[PredictionResult] = []
        # Iterate over each prediction
        for i, (pred, pred_proba) in enumerate(zip(predictions, predicted_probabilities)):
            # Create prediction result
            prediction_enum = PredictionEnum.DEFAULT if pred else PredictionEnum.NO_DEFAULT       
            prediction_result = PredictionResult(
                prediction=prediction_enum,
                probabilities=PredictedProbabilities(
                    default=float(pred_proba[1]),
                    no_default=float(pred_proba[0])
                )
            )
            results.append(prediction_result)

            # Log single prediction record for model monitoring (including batch metadata)
            prediction_monitoring_record = {
                **batch_metadata,
                "prediction_id": str(uuid.uuid4()),
                "inputs": pipeline_input_dict_ls[i],
                "prediction": prediction_enum.value,
                "probabilities": {
                    "default": float(pred_proba[1]),
                    "no_default": float(pred_proba[0])
                },
            }
            monitoring_logger.info(json.dumps(prediction_monitoring_record))  # converts record to JSON string for log

        return PredictionResponse(results=results)

    except Exception as e:
        logger.error("Error during predict: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during loan default prediction")
