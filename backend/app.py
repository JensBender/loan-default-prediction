# --- Imports ---
# Standard library imports
import logging
from pathlib import Path
from typing import List, Dict, Any

# Third-party library imports
from fastapi import FastAPI, HTTPException
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

# --- Logger ---
# Setup a named logger "backend.app" to distinguish logs from frontend
logger = logging.getLogger(__name__)

# --- Helper Functions ---
# Function to load a pre-trained scikit-learn pipeline from local machine
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


# Function to download and load a pipeline from a Hugging Face Hub repository
def load_pipeline_from_huggingface(repo_id: str, filename: str) -> Pipeline:
    try:
        logger.info("Downloading pipeline from Hugging Face Hub...")
        # hf_hub_download returns the path to the downloaded file
        pipeline_path = hf_hub_download(repo_id=repo_id, filename=filename)
        logger.info(f"Successfully downloaded pipeline '{filename}' from Hugging Face Hub repository '{repo_id}'.")
        # Load the pipeline from the downloaded file
        pipeline = load_pipeline_from_local(pipeline_path)
        logger.info(f"Successfully loaded pipeline from '{pipeline_path}' in Docker container.")
        return pipeline
    except Exception as e:
        raise RuntimeError(f"Error when loading pipeline '{filename}' from Hugging Face Hub repo '{repo_id}'") from e 


# --- ML Pipeline ---
# Load loan default prediction pipeline (including data preprocessing and Random Forest Classifier model) from Hugging Face Hub
pipeline = load_pipeline_from_huggingface(
    repo_id="JensBender/loan-default-prediction-pipeline", 
    filename="loan_default_rf_pipeline.joblib"
)

# Load pipeline from local machine
# root_dir = get_root_directory()  # get path to root directory
# pipeline_path = root_dir / "models" / "loan_default_rf_pipeline.joblib"  # get path to pipeline file
# pipeline = load_pipeline_from_local(pipeline_path)

# --- API ---
# Create FastAPI app
app = FastAPI()


# Prediction endpoint 
@app.post("/predict", response_model=PredictionResponse)
def predict(pipeline_input: PipelineInput | List[PipelineInput]) -> PredictionResponse:  # JSON object -> PipelineInput | JSON array -> List[PipelineInput]
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

        return PredictionResponse(results=results)

    except Exception as e:
        logger.error("Error during predict: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during loan default prediction")
