# Imports
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress deprecation warnings
import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # add the parent directory to the path
from app.app import check_missing_values, validate_data_types


# Define valid input values for testing as dictionary
@pytest.fixture
def valid_inputs():
    return {
        "age": 30,
        "married": "Married",
        "income": 1000000,
        "car_ownership": "Yes",
        "house_ownership": "Rented",
        "current_house_yrs": 12,
        "city": "Delhi",
        "state": "Assam",
        "profession": "Architect",
        "experience": 10,
        "current_job_yrs": 7
    }


def test_check_no_missing_values(valid_inputs):
    assert check_missing_values(**valid_inputs) == None


@pytest.mark.parametrize("missing_input, expected_error_message", [
    ("age", "Please provide: Age."),
    ("married", "Please provide: Married/Single."),
    ("income", "Please provide: Income."),
    ("car_ownership", "Please provide: Car Ownership."),
    ("house_ownership", "Please provide: House Ownership."),
    ("current_house_yrs", "Please provide: Current House Years."),
    ("city", "Please provide: City."),
    ("state", "Please provide: State."),
    ("profession", "Please provide: Profession."),
    ("experience", "Please provide: Experience."),
    ("current_job_yrs", "Please provide: Current Job Years."),
])
def test_check_single_missing_value(valid_inputs, missing_input, expected_error_message):
    inputs = valid_inputs.copy()
    inputs[missing_input] = None
    assert check_missing_values(**inputs) == expected_error_message


@pytest.mark.parametrize("missing_input_1, missing_input_2, expected_error_message", [
    ("age", "married", "Please provide: Age and Married/Single."),
    ("income", "car_ownership", "Please provide: Income and Car Ownership."),
    ("house_ownership", "current_house_yrs", "Please provide: House Ownership and Current House Years."),
    ("city", "state", "Please provide: City and State."),
    ("profession", "experience", "Please provide: Profession and Experience."),
    ("age", "current_job_yrs", "Please provide: Age and Current Job Years."),
])
def test_check_two_missing_values(valid_inputs, missing_input_1, missing_input_2, expected_error_message):
    inputs = valid_inputs.copy()
    inputs[missing_input_1] = None
    inputs[missing_input_2] = None
    assert check_missing_values(**inputs) == expected_error_message


def test_check_all_missing_values():
    inputs = {
        "age": None,
        "married": None,
        "income": None,
        "car_ownership": None,
        "house_ownership": None,
        "current_house_yrs": None,
        "city": None,
        "state": None,
        "profession": None,
        "experience": None,
        "current_job_yrs": None
    }
    expected_error_message = (
        "Please provide: Age, Married/Single, Income, Car Ownership, House Ownership, Current House Years, "
        "City, State, Profession, Experience and Current Job Years."
        )
    assert check_missing_values(**inputs) == expected_error_message


def test_check_zero_as_missing_value(valid_inputs):
    inputs = valid_inputs.copy()
    inputs["age"] = 0
    assert check_missing_values(**inputs) == None  # zero is not considered a missing value for numerical inputs


# --- Inputs and their value ranges ---
# age
# married ["Single", "Married"]
# income
# car_ownership ["Yes", "No"] 
# house_ownership ["Rented", "Owned", "Neither Rented Nor Owned"] 
# current_house_yrs [10-14]
# city ["Adoni", "Agartala", "Agra", "Ahmedabad", "Ahmednagar", ...]
# state ["Andhra_Pradesh", "Assam", "Bihar", "Chandigarh", "Chhattisgarh", "Delhi", ...]
# profession ["Air_traffic_controller", "Analyst", "Architect", "Army_officer", "Artist", ...]
# experience [0-20]
# current_job_yrs [0-14]


def test_validate_numerical_inputs():
    # No invalid data types
    assert validate_data_types(30, "Married", 1000000, "Yes", "Rented", 12, "Delhi", "Assam", "Architect", 10, 7) == None