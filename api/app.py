# --- Imports ---
# Standard library imports
import os
from typing import List, Dict, Any

# Third-party library imports
from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import joblib
import uvicorn

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
    PredictionEnum,
    PipelineInput,
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
    
    # Load pipeline with error handling
    try:
        pipeline = joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load pipeline from '{path}'") from e
    
    # Ensure pipeline has .predict_proba() method
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError(f"Loaded pipeline does not have a .predict_proba() method")

    return pipeline


# Use function to load the loan default prediction pipeline (including data preprocessing and Random Forest Classifier model)
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
    # Standardize input to List[dict]
    pipeline_input_dict_ls: List[Dict[str, Any]]
    if isinstance(pipeline_input, list):
        pipeline_input_dict_ls = [input.model_dump() for input in pipeline_input]
    else:  # isinstance(pipeline_input, PipelineInput)
        pipeline_input_dict_ls = [pipeline_input.model_dump()]
        
    # Use pipeline to predict probabilities 
    pipeline_input_df: pd.DataFrame = pd.DataFrame(pipeline_input_dict_ls)
    pred_proba_np: np.ndarray = pipeline.predict_proba(pipeline_input_df)

    # Apply optimized threshold to convert probabilities to binary predictions
    optimized_threshold: float = 0.29  # see threshold optimization in training script "loan_default_prediction.ipynb"
    pred_np: np.ndarray = (pred_proba_np[:, 1] >= optimized_threshold)  # bool 1d-array based on class 1 "Default"

    # Create API response 
    results: List[PredictionResult] = []
    for pred, pred_proba in zip(pred_np, pred_proba_np):  
        prediction_enum = PredictionEnum.DEFAULT if pred else PredictionEnum.NO_DEFAULT 
        predicted_probabilities = PredictedProbabilities(default=pred_proba[1], no_default=pred_proba[0])
        prediction_result = PredictionResult(prediction=prediction_enum, probabilities=predicted_probabilities)
        results.append(prediction_result)

    return PredictionResponse(
        n_predictions=len(results),
        results=results
    )


# Launch API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
