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
        "income": 1000000,
        "age": 30,
        "experience": 10,
        "married": "married",
        "house_ownership": "rented",
        "car_ownership": "yes",
        "profession": "architect",
        "city": "delhi_city",
        "state": "assam",
        "current_job_yrs": 7,        
        "current_house_yrs": 12
    }


# --- Pydantic Model: PipelineInput ---
@pytest.mark.unit
def test_pipeline_input_happy_path(valid_pipeline_input: Dict[str, Any]) -> None:
    pipeline_input = PipelineInput(**valid_pipeline_input)
    assert pipeline_input.income == 1000000
    assert pipeline_input.age == 30
    assert pipeline_input.experience == 10
    assert pipeline_input.married == "married"
    assert pipeline_input.house_ownership == "rented"
    assert pipeline_input.car_ownership == "yes"
    assert pipeline_input.profession == "architect"
    assert pipeline_input.city == "delhi_city"
    assert pipeline_input.state == "assam"
    assert pipeline_input.current_job_yrs == 7
    assert pipeline_input.current_house_yrs == 12

@pytest.mark.unit
@pytest.mark.parametrize("field, float_value, expected_int", [
    ("age", 29.5, 30),
    ("income", 1000000.5, 1000000)
])
def test_pipeline_input_converts_float_to_int(valid_pipeline_input: Dict[str, Any], field: str, float_value: float, expected_int: int) -> None:
    pipeline_input_with_float = valid_pipeline_input.copy()
    pipeline_input_with_float[field] = float_value
    pipeline_input = PipelineInput(**pipeline_input_with_float)
    assert getattr(pipeline_input, field) == expected_int
