# Standard library imports
from typing import List, Dict, Any
from enum import Enum

# Third-party library imports
import pytest
from pydantic import ValidationError 

# Local imports
from api.schemas import (
    MarriedEnum,
    HouseOwnershipEnum,
    CarOwnershipEnum,
    ProfessionEnum,
    CityEnum,
    StateEnum,
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
    def test_raises_validation_error_if_required_field_is_missing(
            self, 
            valid_pipeline_input: Dict[str, Any],
            required_field: str
    ) -> None:
        pipeline_input_with_missing_required_field = valid_pipeline_input.copy()
        del pipeline_input_with_missing_required_field[required_field]
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PipelineInput(**pipeline_input_with_missing_required_field)
        errors = exc_info.value.errors()
        # Ensure error location of all errors is the required field we are testing
        assert all(error["loc"][0] == required_field for error in errors)
        # Ensure error type of at least one error is "missing"
        assert any(error["type"] == "missing" for error in errors)

    # Missing optional field
    @pytest.mark.unit 
    @pytest.mark.parametrize("optional_field", OPTIONAL_FIELDS)
    def test_assigns_none_if_optional_field_is_missing(
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
    def test_raises_validation_error_if_required_field_is_none(
            self, 
            valid_pipeline_input: Dict[str, Any],
            required_field: str
    ) -> None:
        pipeline_input_with_missing_required_value = valid_pipeline_input.copy()
        pipeline_input_with_missing_required_value[required_field] = None
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PipelineInput(**pipeline_input_with_missing_required_value)
        errors = exc_info.value.errors()
        # Ensure error location of all errors is the required field we are testing
        assert all(error["loc"][0] == required_field for error in errors)
        # Ensure error type of at least one error is "int_type", "float_type" or "enum" (which take precedence over "none_forbidden")
        assert any(error["type"] in ["int_type", "float_type", "enum"] for error in errors)

    # Missing value in optional field
    @pytest.mark.unit 
    @pytest.mark.parametrize("optional_field", OPTIONAL_FIELDS)
    def test_assigns_none_if_optional_field_is_none(
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
    def test_raises_validation_error_if_string_field_has_wrong_type(
            self, 
            valid_pipeline_input: Dict[str, Any],
            string_field: str, 
            wrong_data_type: Any
    ) -> None:
        pipeline_input_with_wrong_type = valid_pipeline_input.copy()
        pipeline_input_with_wrong_type[string_field] = wrong_data_type
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PipelineInput(**pipeline_input_with_wrong_type)
        errors = exc_info.value.errors()
        # Ensure error location of all errors is the string field we are testing
        assert all(error["loc"][0] == string_field for error in errors)
        # Ensure error type of at least one error is "enum" 
        assert any(error["type"] == "enum" for error in errors)

    # Wrong data type of numeric fields
    @pytest.mark.unit 
    @pytest.mark.parametrize("numeric_field", ["income", "age", "experience", "current_job_yrs", "current_house_yrs"])
    @pytest.mark.parametrize("wrong_data_type", [
        "a string",
        "1.23",
        False,
        ["a", "list"],
        ("a", "tuple"),
        {"a": "dictionary"},
        {"a", "set"}
    ])
    def test_raises_validation_error_if_numeric_field_has_wrong_type(
            self, 
            valid_pipeline_input: Dict[str, Any],
            numeric_field: str, 
            wrong_data_type: Any
    ) -> None:
        pipeline_input_with_wrong_type = valid_pipeline_input.copy()
        pipeline_input_with_wrong_type[numeric_field] = wrong_data_type
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PipelineInput(**pipeline_input_with_wrong_type)
        errors = exc_info.value.errors()
        # Ensure error location of all errors is the numeric field we are testing
        assert all(error["loc"][0] == numeric_field for error in errors)
        # Ensure error type of at least one error is "int_type" or "float_type" (due to strict mode)
        assert any(error["type"] in ["int_type", "float_type"] for error in errors)

    # Out-of-range numeric values
    @pytest.mark.unit 
    @pytest.mark.parametrize("numeric_field, oor_value", [
        ("income", -50),  # negative 
        ("income", -0.01),  # below minimum 
        ("age", -50),  # negative  
        ("age", 0),  # zero
        ("age", 20.99),  # below minimum 
        ("age", 79.01),  # above maximum
        ("age", 1000),  # large number
        ("experience", -50),  # negative
        ("experience", -0.01),  # below minimum
        ("experience", 20.01),  # above minimum
        ("experience", 1000),  # large number
        ("current_job_yrs", -50),  # negative
        ("current_job_yrs", -0.01),  # below minimum
        ("current_job_yrs", 14.01),  # above maximum
        ("current_job_yrs", 1000),  # large number
        ("current_house_yrs", -50),  # negative
        ("current_house_yrs", 0),  # zero
        ("current_house_yrs", 9.99),  # below minimum
        ("current_house_yrs", 14.01),  # above maximum
        ("current_house_yrs", 1000),  # large number
    ])
    def test_raises_validation_error_if_numeric_value_is_out_of_range(
            self, 
            valid_pipeline_input: Dict[str, Any],
            numeric_field: str, 
            oor_value: int | float
    ) -> None:
        pipeline_input_with_oor_value = valid_pipeline_input.copy()
        pipeline_input_with_oor_value[numeric_field] = oor_value
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PipelineInput(**pipeline_input_with_oor_value)
        errors = exc_info.value.errors()
        # Ensure error location of all errors is the numeric field we are testing
        assert all(error["loc"][0] == numeric_field for error in errors)
        # Ensure error type of at least one error is "greater_than_equal" or "less_than_equal"
        assert any(error["type"] in ["greater_than_equal", "less_than_equal"] for error in errors)

    # Boundary numeric values
    @pytest.mark.unit
    @pytest.mark.parametrize("numeric_field, boundary_value", [
        ("income", 0), ("income", 0.0),  # minimum 
        ("age", 21),  ("age", 21.0),  # minimum 
        ("age", 79),  ("age", 79.0),  # maximum
        ("experience", 0),  ("experience", 0.0), 
        ("experience", 20),  ("experience", 20.0), 
        ("current_job_yrs", 0),  ("current_job_yrs", 0.0), 
        ("current_job_yrs", 14),  ("current_job_yrs", 14.0), 
        ("current_house_yrs", 10),  ("current_house_yrs", 10.0), 
        ("current_house_yrs", 14),  ("current_house_yrs", 14.0), 
    ])
    def test_boundary_values_are_valid_for_numeric_fields(
        self, 
        valid_pipeline_input: Dict[str, Any],
        numeric_field: str,
        boundary_value: int | float
    ) -> None:
        pipeline_input_with_boundary_value = valid_pipeline_input.copy()
        pipeline_input_with_boundary_value[numeric_field] = boundary_value
        pipeline_input = PipelineInput(**pipeline_input_with_boundary_value)
        value = getattr(pipeline_input, numeric_field)
        expected_value = int(boundary_value)
        assert isinstance(value, int)
        assert value == expected_value

    # Invalid string enum values
    @pytest.mark.unit 
    @pytest.mark.parametrize("string_field, invalid_string_enum", [
        ("married", "divorced"), 
        ("married", "yes"), 
        ("married", "no"), 
        ("married", "Single"),  # wrong casing
        ("married", "SINGLE"),  # wrong casing
        ("married", "Married"),  # wrong casing
        ("married", "MARRIED"),  # wrong casing
        ("house_ownership", "maybe"), 
        ("house_ownership", "yes"), 
        ("house_ownership", "no"), 
        ("house_ownership", "mortgaged"), 
        ("house_ownership", "hopefully_in_the_future"), 
        ("house_ownership", "Rented"),  # wrong casing
        ("house_ownership", "OWNED"),  # wrong casing
        ("house_ownership", "Norent_Noown"),  # wrong casing
        ("car_ownership", "maybe"), 
        ("car_ownership", "lamborghini"), 
        ("car_ownership", "soon"), 
        ("car_ownership", "Yes"),  # wrong casing
        ("car_ownership", "NO"),  # wrong casing
        ("profession", "unknown"), 
        ("profession", "jedi_knight"), 
        ("profession", "princess"), 
        ("profession", "divorce_lawyer"), 
        ("profession", "Air_Traffic_Controller"),  # wrong casing
        ("profession", "Army_officer"),  # wrong casing
        ("city", "unknown"), 
        ("city", "metropolis"), 
        ("city", "new_york"), 
        ("city", "tokyo"), 
        ("city", "Chandigarh_City"),  # wrong casing
        ("city", "ADONI"),  # wrong casing
        ("state", "unknown"), 
        ("state", "india"), 
        ("state", "california"), 
        ("state", "Andhra_Pradesh"),  # wrong casing 
        ("state", "ASSAM"),  # wrong casing 
    ])
    def test_raises_validation_error_for_invalid_string_enum(
            self, 
            valid_pipeline_input: Dict[str, Any],
            string_field: str, 
            invalid_string_enum: str 
    ) -> None:
        pipeline_input_with_invalid_string = valid_pipeline_input.copy()
        pipeline_input_with_invalid_string[string_field] = invalid_string_enum
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PipelineInput(**pipeline_input_with_invalid_string)
        errors = exc_info.value.errors()
        # Ensure error location of all errors is the string field we are testing
        assert all(error["loc"][0] == string_field for error in errors)
        # Ensure error type of all errors is "enum"
        assert all(error["type"] == "enum" for error in errors)

    # Valid string enum values
    @pytest.mark.unit 
    @pytest.mark.parametrize("string_field, valid_string, expected_enum_member", [
        ("married", "single", MarriedEnum.SINGLE), 
        ("house_ownership", "rented", HouseOwnershipEnum.RENTED),    
        ("car_ownership", "yes", CarOwnershipEnum.YES), 
    ])
    def test_accepts_valid_string_enum(
            self, 
            valid_pipeline_input: Dict[str, Any],
            string_field: str, 
            valid_string: str,
            expected_enum_member: Enum
    ) -> None:
        pipeline_input_with_valid_string = valid_pipeline_input.copy()
        pipeline_input_with_valid_string[string_field] = valid_string
        pipeline_input = PipelineInput(**pipeline_input_with_valid_string)
        enum_member = getattr(pipeline_input, string_field)
        assert enum_member == expected_enum_member
