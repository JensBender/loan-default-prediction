# Standard library imports
import os
import sys
import warnings

# Third-party library imports
import pytest

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  

# Add project root directory to the path for local imports (by going up two levels from current directory)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  

# Local imports
from api.app import (
    PipelineInput,
    PredictedProbabilities,
    PredictionResult,
    PredictionResponse
)


# --- Fixtures ----
# Define valid input as dictionary for testing 
@pytest.fixture
def valid_pipeline_input():
    return {
        "age": 30,
        "married": "married",
        "income": 1000000,
        "car_ownership": "yes",
        "house_ownership": "rented",
        "current_house_yrs": 12,
        "city": "delhi_city",
        "state": "assam",
        "profession": "architect",
        "experience": 10,
        "current_job_yrs": 7        
    }


# --- Pydantic Model Classes ---
@pytest.mark.unit
def test_pipeline_input_happy_path(valid_pipeline_input):
    pipeline_input = PipelineInput(**valid_pipeline_input)
    assert pipeline_input == PipelineInput(
        age=30,
        married="married",
        income=1000000,
        car_ownership="yes",
        house_ownership="rented",
        current_house_yrs=12,
        city="delhi_city",
        state="assam",
        profession="architect",
        experience=10,
        current_job_yrs=7          
    )
    