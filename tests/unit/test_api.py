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
# Happy path
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

# Convert float to int
@pytest.mark.unit
@pytest.mark.parametrize("field, input_value, expected_value", [
    ("income", 100000.6, 100001),  # round up
    ("income", 100000.4, 100000),  # round down
    ("age", 25.5, 26),  # round up for odd numbers
    ("age", 26.5, 26),  # round down for even numbers (banker's rounding)
    ("experience", 10.0, 10),  # whole float
    ("experience", 0.0, 0),  
    ("current_job_yrs", 5, 5),  # passthrough int
    ("current_job_yrs", 14, 14),  
    ("current_house_yrs", 10.2, 10),  # round at lower boundary
    ("current_house_yrs", 13.9, 14),  # round at upper boundary
])
def test_pipeline_input_converts_float_to_int(
    valid_pipeline_input: Dict[str, Any], 
    field: str, 
    input_value: float | int, 
    expected_value: int
) -> None:
    pipeline_input_dict = valid_pipeline_input.copy()
    pipeline_input_dict[field] = input_value
    pipeline_input = PipelineInput(**pipeline_input_dict)
    assert getattr(pipeline_input, field) == expected_value

# Missing required field
# Missing optional field
# Missing values in required field
# Missing value in optional field
# Wrong type
# Out-of-range numeric value
# Invalid string enum value
