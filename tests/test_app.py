# Imports
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress deprecation warnings
import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # add the parent directory to the path
from app.app import strip_whitespace, check_missing_values, validate_data_types, check_out_of_range_values


# Define valid input as dictionary for testing 
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


# --- Test strip_whitespace() function ---
# Remove various types of whitespace in string inputs
def test_strip_whitespace():
    whitespace_inputs = {
        "age": 30,
        "married": " Married",
        "income": 1000000,
        "car_ownership": " Yes  ",
        "house_ownership": "Rented\t",
        "current_house_yrs": 12,
        "city": "\nDelhi",
        "state": " Assam \t \n ",
        "profession": "\t\t Architect ",
        "experience": 10,
        "current_job_yrs": 7
    }
    expected_cleaned_inputs = {
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
    cleaned_inputs = strip_whitespace(whitespace_inputs)
    assert cleaned_inputs == expected_cleaned_inputs


# Ensure clean inputs remain unchanged by strip_whitespace()
def test_strip_whitespace_clean_inputs_remain_unchanged(valid_inputs):
    cleaned_inputs = strip_whitespace(valid_inputs)
    assert cleaned_inputs == valid_inputs


# --- Test check_missing_values() function ---
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


# Ensure 0 is a valid numerical input that does not trigger an error
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


# --- Test validate_data_types() function ---
# No invalid data types
def test_no_invalid_data_types(valid_inputs):
    assert validate_data_types(valid_inputs) == None


# All invalid data types
def test_invalid_datatype_message_for_all_inputs():
    inputs = {
        "age": "invalid string",
        "married": 12345,
        "income": ["invalid", "list"],
        "car_ownership": 1,
        "house_ownership": 0.0,
        "current_house_yrs": ("invalid", "tuple"),
        "city": {"invalid": "dictionary"},
        "state": True,
        "profession": 12345.67,
        "experience": "invalid string",
        "current_job_yrs": ["invalid", "list"]
    }
    expected_error_message = (
        "Data type error! Age, Income, Current House Years, Experience and Current Job Years must be numbers."
        "Married/Single, House Ownership, Car Ownership, Profession, City and State must be strings."
    )
    assert validate_data_types(inputs) == expected_error_message


# Single invalid data type in a numerical input
@pytest.mark.parametrize("invalid_numerical_data_type", ["invalid string", ["invalid", "list"], ("invalid", "tuple"), {"invalid": "dictionary"}])
def test_invalid_datatype_message_for_single_numerical_input(valid_inputs, invalid_numerical_data_type):
    inputs = valid_inputs.copy()
    inputs["age"] = invalid_numerical_data_type  # age represents all numerical inputs
    assert validate_data_types(inputs) == "Data type error! Age must be a number."


# Single invalid data type in a string input
@pytest.mark.parametrize("invalid_string_data_type", [123, 123.45, False, ["invalid", "list"], ("invalid", "tuple"), {"invalid": "dictionary"}])
def test_invalid_datatype_message_for_single_string_input(valid_inputs, invalid_string_data_type):
    inputs = valid_inputs.copy()
    inputs["married"] = invalid_string_data_type  # married represents all string inputs
    assert validate_data_types(inputs) == "Data type error! Married/Single must be a string."


# --- Test check_out_of_range_values() function ---
# No out-of-range values
def test_no_out_of_range_values(valid_inputs):
    assert check_out_of_range_values(valid_inputs) == None


# All out-of-range values
def test_error_message_for_all_out_of_range_values(valid_inputs):
    inputs = valid_inputs.copy()
    inputs["age"] = 150  # Out of range age
    inputs["income"] = -1000  # Out of range income
    inputs["current_house_yrs"] = 20  # Out of range current house years
    inputs["experience"] = -5  # Out of range experience
    inputs["current_job_yrs"] = 15  # Out of range current job years
    expected_error_message = (
        "Out-of-range value error! The system is designed for applicants with age 21-79, a non-negative income, "
        "10-14 current house years, 0-20 years of experience and 0-14 current job years."
    )
    assert check_out_of_range_values(inputs) == expected_error_message


# Age out-of-range
@pytest.mark.parametrize("age_value, expected_error_message", [
    (-50, "Out-of-range value error! The system is designed for applicants with age 21-79."), 
    (0, "Out-of-range value error! The system is designed for applicants with age 21-79."), 
    (20, "Out-of-range value error! The system is designed for applicants with age 21-79."), 
    (21, None), 
    (79, None), 
    (80, "Out-of-range value error! The system is designed for applicants with age 21-79."),
    (1000, "Out-of-range value error! The system is designed for applicants with age 21-79.")
])
def test_error_message_for_age_out_of_range(valid_inputs, age_value, expected_error_message):
    inputs = valid_inputs.copy()
    inputs["age"] = age_value
    assert check_out_of_range_values(inputs) == expected_error_message


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