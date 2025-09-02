# --- Imports ---
# Standard library imports
from enum import Enum
from typing import List, Annotated
import math

# Third-party library imports
from pydantic import BaseModel, Field, field_validator, model_validator

# Local imports
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

# --- Custom Data Types ---
# Annotate and combine existing types with custom constraints (for Pydantic data model)
Income = Annotated[int, INCOME_CONSTRAINTS] | Annotated[float, INCOME_CONSTRAINTS]
Age = Annotated[int, AGE_CONSTRAINTS] | Annotated[float, AGE_CONSTRAINTS]
Experience = Annotated[int, EXPERIENCE_CONSTRAINTS] | Annotated[float, EXPERIENCE_CONSTRAINTS]
CurrentJobYrs =  Annotated[int, CURRENT_JOB_YRS_CONSTRAINTS] | Annotated[float, CURRENT_JOB_YRS_CONSTRAINTS]
CurrentHouseYrs = Annotated[int, CURRENT_HOUSE_YRS_CONSTRAINTS] | Annotated[float, CURRENT_HOUSE_YRS_CONSTRAINTS]

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
    def convert_float_to_int(cls, value: float | int) -> int:
        if isinstance(value, float):
            return int(round(value))
        return value


# Predicted probabilities model
class PredictedProbabilities(BaseModel):
    default: float = Field(..., ge=0.0, le=1.0, serialization_alias="Default")
    no_default: float = Field(..., ge=0.0, le=1.0, serialization_alias="No Default")

    @field_validator("default", "no_default")
    def round_to_3_decimals(cls, value: float) -> float:
        return round(value, 3)

    @model_validator(mode="after")  # happens after rounding
    def check_probabilities_sum_to_one(self) -> "PredictedProbabilities":
        if not math.isclose(self.default + self.no_default, 1.0, abs_tol=0.001):
            raise ValueError("Probabilities must sum to 1.0")
        return self


# Prediction result model
class PredictionResult(BaseModel):
    prediction: PredictionEnum 
    probabilities: PredictedProbabilities


# Prediction response model
class PredictionResponse(BaseModel):
    n_predictions: int 
    results: List[PredictionResult]