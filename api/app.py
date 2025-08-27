# --- Imports ---
# Standard library imports
import os
from enum import Enum
from typing import List, Annotated

# Third-party library imports
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator 
import pandas as pd
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
from app.global_constants import (
    MARRIED_LABELS,
    HOUSE_OWNERSHIP_LABELS,
    CAR_OWNERSHIP_LABELS,
    PROFESSION_LABELS,
    CITY_LABELS,
    STATE_LABELS
)

# --- Constants ---
# Input constraints (for Pydantic data model)
INCOME_CONSTRAINTS = Field(strict=True, ge=0)
AGE_CONSTRAINTS = Field(strict=True, ge=21, le=79)
EXPERIENCE_CONSTRAINTS = Field(strict=True, ge=0, le=20)
CURRENT_JOB_YRS_CONSTRAINTS = Field(strict=True, ge=0, le=14)
CURRENT_HOUSE_YRS_CONSTRAINTS = Field(strict=True, ge=10, le=14)

# --- Enums ---
# Custom Enum classes for string inputs based on global constants (for Pydantic data model)
MarriedEnum = Enum("MarriedEnum", {label.upper(): label for label in MARRIED_LABELS}, type=str)
HouseOwnershipEnum = Enum("HouseOwnershipEnum", {label.upper(): label for label in HOUSE_OWNERSHIP_LABELS}, type=str)
CarOwnershipEnum = Enum("CarOwnershipEnum", {label.upper(): label for label in CAR_OWNERSHIP_LABELS}, type=str)
ProfessionEnum = Enum("ProfessionEnum", {label.upper(): label for label in PROFESSION_LABELS}, type=str)
CityEnum = Enum("CityEnum", {label.upper(): label for label in CITY_LABELS}, type=str)
StateEnum = Enum("StateEnum", {label.upper(): label for label in STATE_LABELS}, type=str)


# Enum for possible prediction strings
class PredictionEnum(str, Enum):
    DEFAULT = "Default"
    NO_DEFAULT = "No Default"
  
    
# --- Pydantic Data Models ---
# Custom data types for validation (that annotate and combine existing types with custom constraints)
Income = Annotated[int, INCOME_CONSTRAINTS] | Annotated[float, INCOME_CONSTRAINTS]
Age = Annotated[int, AGE_CONSTRAINTS] | Annotated[float, AGE_CONSTRAINTS]
Experience = Annotated[int, EXPERIENCE_CONSTRAINTS] | Annotated[float, EXPERIENCE_CONSTRAINTS]
CurrentJobYrs =  Annotated[int, CURRENT_JOB_YRS_CONSTRAINTS] | Annotated[float, CURRENT_JOB_YRS_CONSTRAINTS]
CurrentHouseYrs = Annotated[int, CURRENT_HOUSE_YRS_CONSTRAINTS] | Annotated[float, CURRENT_HOUSE_YRS_CONSTRAINTS]


# Pipeline input model
class PipelineInput(BaseModel):
    income: Income
    age: Age
    experience: Experience
    married: MarriedEnum | None = None 
    house_ownership: HouseOwnershipEnum | None = None 
    car_ownership: CarOwnershipEnum | None = None 
    profession: ProfessionEnum  
    city: CityEnum 
    state: StateEnum 
    current_job_yrs: CurrentJobYrs
    current_house_yrs: CurrentHouseYrs

    @field_validator("income", "age", "experience", "current_job_yrs", "current_house_yrs")
    def convert_float_to_int(cls, value):
        if isinstance(value, float):
            return int(round(value))
        return value


# Predicted probabilities model
class PredictedProbabilities(BaseModel):
    default: float = Field(..., serialization_alias="Default")
    no_default: float = Field(..., serialization_alias="No Default")

    @field_validator("default", "no_default")
    def round_to_3_decimals(cls, value: float) -> float:
        return round(value, 3)


# Prediction result model
class PredictionResult(BaseModel):
    prediction: PredictionEnum 
    probabilities: PredictedProbabilities


# Prediction response model
class PredictionResponse(BaseModel):
    n_predictions: int 
    results: List[PredictionResult]


# --- Pipeline ---
# Load the pre-trained ML pipeline to predict loan default (including data preprocessing and Random Forest Classifier model)
pipeline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "loan_default_rf_pipeline.joblib")
pipeline = joblib.load(pipeline_path)

# --- API ---
# Create FastAPI app
app = FastAPI()


# Prediction endpoint 
@app.post("/predict", response_model=PredictionResponse)
def predict(pipeline_input: PipelineInput | List[PipelineInput]):  # JSON object -> PipelineInput | JSON array -> List[PipelineInput]
    # Standardize input to List[dict]
    if isinstance(pipeline_input, list):
        pipeline_input_dict_ls = [input.model_dump() for input in pipeline_input]
    else:  # isinstance(pipeline_input, PipelineInput)
        pipeline_input_dict_ls = [pipeline_input.model_dump()]
        
    # Use pipeline to predict probabilities 
    pipeline_input_df = pd.DataFrame(pipeline_input_dict_ls)
    pred_proba_np = pipeline.predict_proba(pipeline_input_df)

    # Apply optimized threshold to convert probabilities to binary predictions
    optimized_threshold = 0.29  # see threshold optimization in training script "loan_default_prediction.ipynb"
    pred_np = (pred_proba_np[:, 1] >= optimized_threshold)  # bool 1d-array based on class 1 "Default"

    # Create API response 
    results = []
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
