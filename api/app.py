# --- Imports ---
# Standard library imports
import os
from typing import List, Dict, Any

# Third-party library imports
from fastapi import FastAPI
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import joblib

# Local imports
from app.custom_transformers import (
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
from api.schemas import (
    PipelineInput,
    PredictionEnum,
    PredictedProbabilities,
    PredictionResult,
    PredictionResponse    
)


# --- ML Pipeline ---
# Function to load a pre-trained scikit-learn pipeline
def load_pipeline(path: str) -> Pipeline:
    # Ensure file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pipeline file not found at '{path}'")
    
    # Load pipeline 
    try:
        pipeline = joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load pipeline from '{path}'") from e
    
    # Ensure loaded object is a scikit-learn Pipeline
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Loaded object is not a scikit-learn Pipeline")

    # Ensure pipeline has .predict_proba() method
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError(f"Loaded pipeline does not have a .predict_proba() method")

    return pipeline


# Load loan default prediction pipeline (including data preprocessing and Random Forest Classifier model)
pipeline_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "..", 
    "models", 
    "loan_default_rf_pipeline.joblib"
)
pipeline = load_pipeline(path=pipeline_path)

# --- API ---
# Create FastAPI app
app = FastAPI()


# Prediction endpoint 
@app.post("/predict", response_model=PredictionResponse)
def predict(pipeline_input: PipelineInput | List[PipelineInput]) -> PredictionResponse:  # JSON object -> PipelineInput | JSON array -> List[PipelineInput]
    # Standardize input
    pipeline_input_dict_ls: List[Dict[str, Any]]
    if isinstance(pipeline_input, list):
        pipeline_input_dict_ls = [input.model_dump() for input in pipeline_input]
    else:  # isinstance(pipeline_input, PipelineInput)
        pipeline_input_dict_ls = [pipeline_input.model_dump()]
    pipeline_input_df: pd.DataFrame = pd.DataFrame(pipeline_input_dict_ls)
        
    # Use pipeline to predict probabilities 
    predicted_probabilities: np.ndarray = pipeline.predict_proba(pipeline_input_df)

    # Apply optimized threshold to convert probabilities to binary predictions
    optimized_threshold: float = 0.29  # see threshold optimization in training script "loan_default_prediction.ipynb"
    predictions: np.ndarray = (predicted_probabilities[:, 1] >= optimized_threshold)  # bool 1d-array based on class 1 "Default"

    # Create API response 
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
