# --- Imports ---
# Standard library imports
import logging
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

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
# Setup a logger for the backend 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Setup a dedicated logger for monitoring prediction records
monitoring_logger = logging.getLogger("monitoring")
monitoring_logger.setLevel(logging.INFO)
monitoring_logger.propagate = False  # prevents logs from propagating to the backend logger so they don't show up in the console
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
file_handler = logging.FileHandler(log_dir / "prediction_logs.jsonl")  # use JSON lines format, where each line is a JSON object
formatter = logging.Formatter("%(message)s")  # just output the message which will be pre-formatted JSON
file_handler.setFormatter(formatter)
if not monitoring_logger.handlers:  # add the handler to the logger if it doesn't have one already
    monitoring_logger.addHandler(file_handler)


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
def load_pipeline_from_huggingface(repo_id: str, filename: str) -> Pipeline:
    try:
        # .hf_hub_download() downloads the pipeline file and returns its local file path (inside the Docker container)
        # if the pipeline file was already downloaded, it checks for new version on Hugging Face Hub
        # if new version, it will download and use it, else it will use the cached pipeline that is already stored inside the Docker container 
        logger.info(
            f"Connecting to Hugging Face Hub to check for the latest pipeline file '{filename}' in repo '{repo_id}'. "
            "If already cached and up to date, will use local copy."
        )
        pipeline_path = hf_hub_download(repo_id=repo_id, filename=filename)
        # Load pipeline from file inside the Docker container
        pipeline = load_pipeline_from_local(pipeline_path)
        return pipeline
    except Exception as e:
        raise RuntimeError(f"Error loading pipeline '{filename}' from Hugging Face Hub repository '{repo_id}': {e}") from e 


# --- ML Pipeline ---
# Load loan default prediction pipeline (including data preprocessing and Random Forest Classifier model) from Hugging Face Hub
pipeline = load_pipeline_from_huggingface(
    repo_id="JensBender/loan-default-prediction-pipeline", 
    filename="loan_default_rf_pipeline.joblib"
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
        pipeline_input_df: pd.DataFrame = pd.DataFrame(pipeline_input_dict_ls)
            
        # Use pipeline to predict probabilities 
        predicted_probabilities: np.ndarray = pipeline.predict_proba(pipeline_input_df)

        # Apply optimized threshold to convert probabilities to binary predictions
        optimized_threshold: float = 0.29  # see threshold optimization in training script "loan_default_prediction.ipynb"
        predictions: np.ndarray = (predicted_probabilities[:, 1] >= optimized_threshold)  # bool 1d-array based on class 1 "Default"

        # Create prediction response 
        results: List[PredictionResult] = []
        for pred, pred_proba in zip(predictions, predicted_probabilities):  
            prediction_enum = PredictionEnum.DEFAULT if pred else PredictionEnum.NO_DEFAULT 
            prediction_result = PredictionResult(
                prediction=prediction_enum, 
                probabilities=PredictedProbabilities(
                    default=pred_proba[1], 
                    no_default=pred_proba[0]
                )
            )
            results.append(prediction_result)

        # Log prediction ID, timestamp, pipeline version, pipeline input, prediction response, prediction latency, client IP address, and User-Agent for model monitoring
        # Create a unique prediction ID
        prediction_id = str(uuid.uuid4())
        # Get timestamp
        timestamp = datetime.now(timezone.utc).isoformat()
        # Get pipeline_version (hardcoded for now)
        pipeline_version = "1.0"
        # Get prediction latency in milliseconds (hardcoded for now)
        prediction_latency_ms = 100
        # Get client IP address from request headers 
        client_ip = request.headers.get("x-forwared-for", request.client.host)
        # Get User-Agent from request headers dictionary (default to "unknown") 
        user_agent = request.headers.get("user-agent", "unknown")

        return PredictionResponse(results=results)

    except Exception as e:
        logger.error("Error during predict: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during loan default prediction")
