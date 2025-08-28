# Standard library imports
from typing import Dict, Any

# Third-party library imports
import pytest

# Local imports
from api.app import (
    PipelineInput,
    PredictedProbabilities,
    PredictionResult,
    PredictionResponse
)


# --- Fixtures ----
# Define valid pipeline input for testing 
@pytest.fixture
def valid_pipeline_input() -> Dict[str, Any]:
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


# --- Pydantic Model: PipelineInput ---
@pytest.mark.unit
def test_pipeline_input_happy_path(valid_pipeline_input: Dict[str, Any]) -> None:
    pipeline_input = PipelineInput(**valid_pipeline_input)
    assert pipeline_input.age == 30
    assert pipeline_input.married == "married"
    assert pipeline_input.income == 1000000
    assert pipeline_input.car_ownership == "yes"
    assert pipeline_input.house_ownership == "rented"
    assert pipeline_input.current_house_yrs == 12
    assert pipeline_input.city == "delhi_city"
    assert pipeline_input.state == "assam"
    assert pipeline_input.profession == "architect"
    assert pipeline_input.experience == 10
    assert pipeline_input.current_job_yrs == 7
