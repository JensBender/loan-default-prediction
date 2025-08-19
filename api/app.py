# Standard library imports
import os
import pickle
from enum import Enum

# Third-party library imports
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, StrictInt, StrictFloat
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
    CAR_OWNERSHIP_LABELS,
    HOUSE_OWNERSHIP_LABELS,
    PROFESSION_LABELS,
    CITY_LABELS,
    STATE_LABELS
)

# --- Enums ---
# Create custom Enum classes for string inputs from global constants (for Pydantic data validation)
MarriedEnum = Enum("MarriedEnum", {label.upper(): label for label in MARRIED_LABELS}, type=str)
CarOwnershipEnum = Enum("CarOwnershipEnum", {label.upper(): label for label in CAR_OWNERSHIP_LABELS}, type=str)
HouseOwnershipEnum = Enum("HouseOwnershipEnum", {label.upper(): label for label in HOUSE_OWNERSHIP_LABELS}, type=str)
ProfessionEnum = Enum("ProfessionEnum", {label.upper(): label for label in PROFESSION_LABELS}, type=str)
CityEnum = Enum("CityEnum", {label.upper(): label for label in CITY_LABELS}, type=str)
StateEnum = Enum("StateEnum", {label.upper(): label for label in STATE_LABELS}, type=str)

# --- Pipeline ---
# Load the pre-trained ML pipeline to predict loan default (including data preprocessing and Random Forest Classifier model)
pipeline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "loan_default_rf_pipeline.pkl")
with open(pipeline_path, "rb") as file:
    pipeline = pickle.load(file)

# --- API ---
# Create FastAPI app
app = FastAPI()


# --- Pydantic Data Model ---
# Pipeline input
class PipelineInput(BaseModel):
    age: StrictInt | StrictFloat
    married: MarriedEnum | None = None 
    income: StrictInt | StrictFloat
    car_ownership: CarOwnershipEnum | None = None 
    house_ownership: HouseOwnershipEnum | None = None 
    current_house_yrs: StrictInt | StrictFloat
    city: CityEnum 
    state: StateEnum 
    profession: ProfessionEnum  
    experience: StrictInt | StrictFloat
    current_job_yrs: StrictInt | StrictFloat


# --- API Endpoints ---
# Single prediction
@app.post("/predict")
def single_predict(pipeline_input: PipelineInput):
    pipeline_input_df = pd.DataFrame([pipeline_input.model_dump])


# Batch prediction
@app.post("/batch-predict")
def batch_predict():
    pass


# Launch API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
