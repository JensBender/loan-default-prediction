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


# --- check_missing_values() function ---
# No missing inputs
def test_no_missing_values(valid_inputs):
    assert check_missing_values(valid_inputs) == None

# All missing inputs
def test_error_message_for_all_values_missing():
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
    assert check_missing_values(inputs) == expected_error_message


# Single missing numerical input: `if numerical_input in [None, "", [], {}, ()]`
@pytest.mark.parametrize("missing_value_type", [None, "", [], {}, ()])
@pytest.mark.parametrize("numerical_input, expected_partial_error_message", [
    ("age", "Age"),
    ("income", "Income"),
    ("current_house_yrs", "Current House Years"),
    ("experience", "Experience"),
    ("current_job_yrs", "Current Job Years"),
])
def test_error_message_for_single_missing_numerical_input(valid_inputs, missing_value_type, numerical_input, expected_partial_error_message):
    inputs = valid_inputs.copy()
    inputs[numerical_input] = missing_value_type
    error_message = check_missing_values(inputs)
    # Check presence of error message
    assert error_message is not None, f"Expected an error message for {numerical_input}='{missing_value_type}'."
    # Check exact error message text
    assert error_message == f"Please provide: {expected_partial_error_message}.", f"Expected exact error message: 'Please provide: {expected_partial_error_message}.' for {numerical_input}='{missing_value_type}'"


# 0 is a valid numerical input
@pytest.mark.parametrize("numerical_input", ["age", "income", "current_house_yrs", "experience", "current_job_yrs"])
def test_zero_is_valid_numerical_input(valid_inputs, numerical_input):
    inputs = valid_inputs.copy()
    inputs[numerical_input] = 0
    assert check_missing_values(inputs) == None  


# Single missing string input: `if not string_input`, which catches None, "", [], {}, (), 0, 0.0, False
@pytest.mark.parametrize("missing_value_type", [None, "", [], {}, (), 0, 0.0, False])
@pytest.mark.parametrize("string_input, expected_partial_error_message", [
    ("married", "Married/Single"),
    ("car_ownership", "Car Ownership"),
    ("house_ownership", "House Ownership"),
    ("city", "City"),
    ("state", "State"),
    ("profession", "Profession"),
])
def test_error_message_for_single_missing_string_input(valid_inputs, missing_value_type, string_input, expected_partial_error_message):
    inputs = valid_inputs.copy()
    inputs[string_input] = missing_value_type
    error_message = check_missing_values(inputs)
    # Check presence of error message
    assert error_message is not None, f"Expected an error message for {string_input}='{missing_value_type}'"
    # Check exact error message text
    assert error_message == f"Please provide: {expected_partial_error_message}.", f"Expected exact error message: 'Please provide: {expected_partial_error_message}.' for {string_input}='{missing_value_type}'"


# Two missing inputs
@pytest.mark.parametrize("missing_input_1, missing_input_2, expected_error_message", [
    ("age", "married", "Please provide: Age and Married/Single."),
    ("income", "car_ownership", "Please provide: Income and Car Ownership."),
    ("house_ownership", "current_house_yrs", "Please provide: House Ownership and Current House Years."),
    ("city", "state", "Please provide: City and State."),
    ("profession", "experience", "Please provide: Profession and Experience."),
    ("age", "current_job_yrs", "Please provide: Age and Current Job Years."),
])
def test_error_message_for_two_missing_inputs(valid_inputs, missing_input_1, missing_input_2, expected_error_message):
    inputs = valid_inputs.copy()
    inputs[missing_input_1] = None
    inputs[missing_input_2] = None
    assert check_missing_values(inputs) == expected_error_message


# Three missing inputs
@pytest.mark.parametrize("missing_input_1, missing_input_2, missing_input_3, expected_error_message", [
    ("age", "married", "income", "Please provide: Age, Married/Single and Income."),
    ("car_ownership", "house_ownership", "current_house_yrs", "Please provide: Car Ownership, House Ownership and Current House Years."),
    ("city", "state", "profession", "Please provide: City, State and Profession."),
    ("age", "experience", "current_job_yrs", "Please provide: Age, Experience and Current Job Years."),
])
def test_error_message_for_three_missing_inputs(valid_inputs, missing_input_1, missing_input_2, missing_input_3, expected_error_message):
    inputs = valid_inputs.copy()
    inputs[missing_input_1] = None
    inputs[missing_input_2] = None
    inputs[missing_input_3] = None
    assert check_missing_values(inputs) == expected_error_message


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


# --- validate_data_types() function ---
# No invalid data types
def test_no_invalid_data_types(valid_inputs):
    assert validate_data_types(**valid_inputs) == None