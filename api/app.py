# --- Imports ---
# Standard library imports
import os
import pickle
from enum import Enum
from typing import List

# Third-party library imports
from fastapi import FastAPI
from pydantic import BaseModel, Field, StrictInt, StrictFloat, field_validator 
import pandas as pd
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
MarriedEnum = Enum("MarriedEnum", {label.upper(): label for label in MARRIED_LABELS})
CarOwnershipEnum = Enum("CarOwnershipEnum", {label.upper(): label for label in CAR_OWNERSHIP_LABELS})
HouseOwnershipEnum = Enum("HouseOwnershipEnum", {label.upper(): label for label in HOUSE_OWNERSHIP_LABELS})
ProfessionEnum = Enum("ProfessionEnum", {label.upper(): label for label in PROFESSION_LABELS})
CityEnum = Enum("CityEnum", {label.upper(): label for label in CITY_LABELS})
StateEnum = Enum("StateEnum", {label.upper(): label for label in STATE_LABELS})


# --- Pydantic Data Model ---
# Pipeline input
class PipelineInput(BaseModel):
    age: StrictInt | StrictFloat = Field(..., ge=21, le=79)
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

    @field_validator("age", "income", "current_house_yrs", "experience", "current_job_yrs")
    def convert_float_to_int(cls, value):
        if isinstance(value, float):
            return int(round(value))
        return value


# --- Pipeline ---
# Load the pre-trained ML pipeline to predict loan default (including data preprocessing and Random Forest Classifier model)
pipeline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "loan_default_rf_pipeline.pkl")
with open(pipeline_path, "rb") as file:
    pipeline = pickle.load(file)

# --- API ---
# Create FastAPI app
app = FastAPI()


# Prediction endpoint 
@app.post("/predict")
def predict(pipeline_input: PipelineInput | List[PipelineInput]):  # JSON object -> PipelineInput | JSON array -> List[PipelineInput]
    # Standardize input to list of dictionaries
    if isinstance(pipeline_input, list):
        pipeline_input_dict_ls = [input.model_dump() for input in pipeline_input]
    else:  # isinstance(pipeline_input, PipelineInput)
        pipeline_input_dict_ls = [pipeline_input.model_dump()]
        
    # Use pipeline to predict probabilities 
    pipeline_input_df = pd.DataFrame(pipeline_input_dict_ls)
    predicted_probabilities = pipeline.predict_proba(pipeline_input_df)

    # Apply optimized threshold to convert probabilities to binary predictions
    optimized_threshold = 0.29  # see threshold optimization in training script "loan_default_prediction.ipynb"
    predictions = (predicted_probabilities[:, 1] >= optimized_threshold)  # bool 1d-array based on class 1 "Default"

    # Create API response 
    results = []
    for prediction, predicted_probability in zip(predictions, predicted_probabilities):
        results.append({
            "prediction": "Default" if prediction else "No Default",
            "probabilities": {
                "Default": predicted_probability[1],  # Class 1 is "Default"
                "No Default": predicted_probability[0]  # Class 0 is "No Default"
            }
        })

    return {
        "n_predictions": len(results),
        "results": results
    }


# Launch API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
