# Standard library imports
from typing import List, Dict, Any

# Third-party library imports
import pytest
from pydantic import ValidationError 

# Local imports
from api.app import (
    PipelineInput,
    PredictedProbabilities,
    PredictionResult,
    PredictionResponse
)

# --- Constants ----
REQUIRED_FIELDS: List[str] = [
    "income", "age", "experience", "profession", "city", 
    "state", "current_job_yrs", "current_house_yrs"
]
OPTIONAL_FIELDS: List[str] = ["married", "house_ownership", "car_ownership"]


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
class TestPipelineInput:
    # Happy path
    @pytest.mark.unit
    def test_happy_path(self, valid_pipeline_input: Dict[str, Any]) -> None:
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
    def test_converts_float_to_int(
        self,
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
    @pytest.mark.unit 
    @pytest.mark.parametrize("required_field", REQUIRED_FIELDS)
    def test_raises_validation_error_for_missing_required_field(
            self, 
            valid_pipeline_input: Dict[str, Any],
            required_field: str
    ) -> None:
        pipeline_input_with_missing_required_field = valid_pipeline_input.copy()
        del pipeline_input_with_missing_required_field[required_field]
        with pytest.raises(ValidationError):
            PipelineInput(**pipeline_input_with_missing_required_field)

    # Missing optional field
    @pytest.mark.unit 
    @pytest.mark.parametrize("optional_field", OPTIONAL_FIELDS)
    def test_assigns_none_for_missing_optional_field(
            self, 
            valid_pipeline_input: Dict[str, Any],
            optional_field: str
    ) -> None:
        pipeline_input_with_missing_optional_field = valid_pipeline_input.copy()
        del pipeline_input_with_missing_optional_field[optional_field]
        pipeline_input = PipelineInput(**pipeline_input_with_missing_optional_field)
        assert hasattr(pipeline_input, optional_field)
        assert pipeline_input.model_dump()[optional_field] is None

    # Missing value in required field
    @pytest.mark.unit 
    @pytest.mark.parametrize("required_field", REQUIRED_FIELDS)
    def test_raises_validation_error_for_missing_required_value(
            self, 
            valid_pipeline_input: Dict[str, Any],
            required_field: str
    ) -> None:
        pipeline_input_with_missing_required_value = valid_pipeline_input.copy()
        pipeline_input_with_missing_required_value[required_field] = None
        with pytest.raises(ValidationError):
            PipelineInput(**pipeline_input_with_missing_required_value)

    # Missing value in optional field
    @pytest.mark.unit 
    @pytest.mark.parametrize("optional_field", OPTIONAL_FIELDS)
    def test_passes_through_none_for_missing_optional_value(
            self, 
            valid_pipeline_input: Dict[str, Any],
            optional_field: str
    ) -> None:
        pipeline_input_with_missing_optional_value = valid_pipeline_input.copy()
        pipeline_input_with_missing_optional_value[optional_field] = None
        pipeline_input = PipelineInput(**pipeline_input_with_missing_optional_value)
        assert pipeline_input.model_dump()[optional_field] is None

    # Wrong data type of string fields
    @pytest.mark.unit 
    @pytest.mark.parametrize("string_field", ["married", "house_ownership", "car_ownership", "profession", "city", "state"])
    @pytest.mark.parametrize("wrong_data_type", [
        1,
        1.23,
        False,
        ["a", "list"],
        ("a", "tuple"),
        {"a": "dictionary"},
        {"a", "set"}
    ])
    def test_raises_validation_error_for_wrong_type_in_string_field(
            self, 
            valid_pipeline_input: Dict[str, Any],
            string_field: str, 
            wrong_data_type: Any
    ) -> None:
        pipeline_input_with_wrong_type = valid_pipeline_input.copy()
        pipeline_input_with_wrong_type[string_field] = wrong_data_type
        with pytest.raises(ValidationError):
            PipelineInput(**pipeline_input_with_wrong_type)

    # Wrong data type of numerical fields
    # Out-of-range numeric value
    # Invalid string enum value
