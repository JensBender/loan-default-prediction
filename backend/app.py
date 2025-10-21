# --- Imports ---
# Standard library imports
from pathlib import Path
from typing import List, Dict, Any
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
            "stream": "ext://sys.stdout",
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


# --- ML Pipeline ---
# Load loan default prediction pipeline (including data preprocessing and Random Forest Classifier model) from Hugging Face Hub
pipeline_version_tag = "v1.0"
pipeline = load_pipeline_from_huggingface(
    repo_id="JensBender/loan-default-prediction-pipeline", 
    filename="loan_default_rf_pipeline.joblib",
    revision=pipeline_version_tag  
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
def predict(pipeline_input: PipelineInput | List[PipelineInput], request: Request) -> PredictionResponse:  # JSON object -> PipelineInput | JSON array -> List[PipelineInput]
    try:
        # Standardize input
        pipeline_input_dict_ls: List[Dict[str, Any]]
        if isinstance(pipeline_input, list):
            if pipeline_input == []:  # handle empty batch input
                return PredictionResponse(results=[])
            pipeline_input_dict_ls = [input.model_dump() for input in pipeline_input]
        else:  # isinstance(pipeline_input, PipelineInput)
            pipeline_input_dict_ls = [pipeline_input.model_dump()]

        # Get metadata for logging
        # From frontend, fall back to backend request header for direct API calls
        client_ip = pipeline_input_dict_ls[0].pop("client_ip", None)
        if client_ip is None:
            x_forwarded_for = request.headers.get("x-forwarded-for")
            client_ip = x_forwarded_for.split(",")[0].strip() if x_forwarded_for else request.client.host
        user_agent = pipeline_input_dict_ls[0].pop("user_agent", None)
        if user_agent is None:
            user_agent = request.headers.get("user-agent", "unknown")

        # Create DataFrame
        pipeline_input_df: pd.DataFrame = pd.DataFrame(pipeline_input_dict_ls)

        # Use pipeline to batch predict probabilities (and measure latency)
        start_time = time.perf_counter()  # use .perf_counter() for latency measurement and .time() for timestamps
        predicted_probabilities: np.ndarray = pipeline.predict_proba(pipeline_input_df)  
        pipeline_prediction_latency_ms = round((time.perf_counter() - start_time) * 1000)  # rounded to milliseconds

        # Apply optimized threshold to convert probabilities to binary predictions
        optimized_threshold: float = 0.29  # see threshold optimization in training script "loan_default_prediction.ipynb"
        predictions: np.ndarray = (predicted_probabilities[:, 1] >= optimized_threshold)  # bool 1d-array based on class 1 "Default"

        # Create batch-level metadata for logging
        batch_metadata = {
            "batch_id": str(uuid.uuid4()),
            "batch_size": len(pipeline_input_dict_ls),
            "batch_timestamp": datetime.now(timezone.utc).isoformat(),
            "batch_latency_ms": pipeline_prediction_latency_ms,
            "avg_prediction_latency_ms": round(pipeline_prediction_latency_ms / len(pipeline_input_dict_ls)) if len(pipeline_input_dict_ls) > 0 else None,
            "pipeline_version": pipeline_version_tag,
        }

        # --- Create prediction response --- 
        results: List[PredictionResult] = []
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
                "client_ip": client_ip,
                "user_agent": user_agent,
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
