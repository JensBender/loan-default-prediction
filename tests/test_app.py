# Imports
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress deprecation warnings
import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # add the parent directory to the path
from app.app import (
    standardize_string, 
    standardize_inputs, 
    check_missing_values, 
    validate_data_types, 
    check_out_of_range_values
)


# Define valid input as dictionary for testing 
@pytest.fixture
def valid_inputs():
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


# --- Test standardize_string() function ---
@pytest.mark.parametrize("input_value, expected_output", [
    ("   leading spaces", "leading_spaces"),
    ("trailing spaces   ", "trailing_spaces"),
    ("  Leading and Trailing Spaces  ", "leading_and_trailing_spaces"),
    ("\tLeading tab and trailing linebreak\n", "leading_tab_and_trailing_linebreak"),
    ("\n\t", ""),
    ("Multiple  Inner   Spaces", "multiple_inner_spaces"),
    ("Title Case", "title_case"),
    ("MiXeD CaSe", "mixed_case"),
    ("", ""),
    ("   ", ""),
    ("innner\ttab", "innner_tab"),
    ("inner\nnewline", "inner_newline"),
    ("with-mixed_chars 123", "with_mixed_chars_123"),
    ("singleword", "singleword"),
    (123, 123),
    (123.45, 123.45),
    (True, True),
    (None, None),
    (["a", "list"], ["a", "list"]),
    (("a", "tuple"), ("a", "tuple")),
    ({"a": "dictionary"}, {"a": "dictionary"}),
])
def test_standardize_string(input_value, expected_output):
    assert standardize_string(input_value) == expected_output


# --- Test standardize_inputs() function ---
# Remove leading/trailing whitespace, convert to lowercase, and replace inner whitespaces with underscores
def test_standardize_inputs():
    raw_inputs = {
        "age": 30,
        "married": " Married  ",
        "income": 1000000,
        "car_ownership": "\t\tYes  ",
        "house_ownership": "Neither \t Rented \n Nor  Owned",
        "current_house_yrs": 12,
        "city": "\nSangli-Miraj_&_Kupwad",
        "state": " Uttar_Pradesh \t \n ",
        "profession": "Hotel_Manager ",
        "experience": 10,
        "current_job_yrs": 7
    }
    expected_standardized_inputs = {
        "age": 30,
        "married": "married",
        "income": 1000000,
        "car_ownership": "yes",
        "house_ownership": "neither_rented_nor_owned",
        "current_house_yrs": 12,
        "city": "sangli_miraj_&_kupwad",
        "state": "uttar_pradesh",
        "profession": "hotel_manager",
        "experience": 10,
        "current_job_yrs": 7
    }
    assert standardize_inputs(raw_inputs) == expected_standardized_inputs


# Ensure standardized inputs remain unchanged by standardize_inputs()
def test_standardized_inputs_remain_unchanged():
    standardized_inputs = {
        "age": 30,
        "married": "married",
        "income": 1000000,
        "car_ownership": "yes",
        "house_ownership": "neither_rented_nor_owned",
        "current_house_yrs": 12,
        "city": "sangli_miraj_&_kupwad",
        "state": "uttar_pradesh",
        "profession": "hotel_manager",
        "experience": 10,
        "current_job_yrs": 7
    }   
    assert standardize_inputs(standardized_inputs) == standardized_inputs


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
def test_error_message_for_all_out_of_range_values():
    inputs = {
        "age": 150,
        "married": "divorced",
        "income": -1000,
        "car_ownership": "maybe",
        "house_ownership": "hopefully_in_the_future",
        "current_house_yrs": 20,
        "city": "gotham_city",
        "state": "wakanda",
        "profession": "jedi_knight",
        "experience": -5,
        "current_job_yrs": 15
    }
    expected_error_message = (
        "Out-of-range value error: age must be 21-79, married must be 'single' or 'married', income must be a non-negative number, "
        "car ownership must be 'yes' or 'no', house ownership must be 'rented', 'owned', or 'norent_noown', "
        "current house years must be 10-14, city must be one of the predefined cities, state must be one of the predefined states, "
        "profession must be one of the predefined professions, experience must be 0-20 years and current job years must be 0-14."
    )
    assert check_out_of_range_values(inputs) == expected_error_message


# Age out-of-range
@pytest.mark.parametrize("age_value, expected_error_message", [
    (-50, "Out-of-range value error: age must be 21-79."), 
    (0, "Out-of-range value error: age must be 21-79."), 
    (20, "Out-of-range value error: age must be 21-79."), 
    (21, None), 
    (79, None), 
    (80, "Out-of-range value error: age must be 21-79."),
    (1000, "Out-of-range value error: age must be 21-79.")
])
def test_error_message_for_age_out_of_range(valid_inputs, age_value, expected_error_message):
    inputs = valid_inputs.copy()
    inputs["age"] = age_value
    assert check_out_of_range_values(inputs) == expected_error_message


# Married out-of-range
@pytest.mark.parametrize("married_value, expected_error_message", [
    ("divorced", "Out-of-range value error: married must be 'single' or 'married'."),
    ("yes", "Out-of-range value error: married must be 'single' or 'married'."),
    ("no", "Out-of-range value error: married must be 'single' or 'married'."),
    ("single", None),
    ("married", None)
])
def test_error_message_for_married_out_of_range(valid_inputs, married_value, expected_error_message):
    inputs = valid_inputs.copy()
    inputs["married"] = married_value
    assert check_out_of_range_values(inputs) == expected_error_message


# Income out-of-range
@pytest.mark.parametrize("income_value, expected_error_message", [
    (-1000, "Out-of-range value error: income must be a non-negative number."), 
    (-50, "Out-of-range value error: income must be a non-negative number."), 
    (-1, "Out-of-range value error: income must be a non-negative number."), 
    (-0.001, "Out-of-range value error: income must be a non-negative number."), 
    (0, None), 
    (1000, None), 
])
def test_error_message_for_income_out_of_range(valid_inputs, income_value, expected_error_message):
    inputs = valid_inputs.copy()
    inputs["income"] = income_value
    assert check_out_of_range_values(inputs) == expected_error_message


# Car ownership out-of-range
@pytest.mark.parametrize("car_ownership_value, expected_error_message", [
    ("maybe", "Out-of-range value error: car ownership must be 'yes' or 'no'."),
    ("lamborghini", "Out-of-range value error: car ownership must be 'yes' or 'no'."),
    ("soon", "Out-of-range value error: car ownership must be 'yes' or 'no'."),
    ("yes", None),
    ("no", None)
])
def test_error_message_for_car_ownership_out_of_range(valid_inputs, car_ownership_value, expected_error_message):
    inputs = valid_inputs.copy()
    inputs["car_ownership"] = car_ownership_value
    assert check_out_of_range_values(inputs) == expected_error_message  


# House ownership out-of-range
@pytest.mark.parametrize("house_ownership_value, expected_error_message", [
    ("maybe", "Out-of-range value error: house ownership must be 'rented', 'owned', or 'norent_noown'."),
    ("yes", "Out-of-range value error: house ownership must be 'rented', 'owned', or 'norent_noown'."),
    ("no", "Out-of-range value error: house ownership must be 'rented', 'owned', or 'norent_noown'."),
    ("mortgaged", "Out-of-range value error: house ownership must be 'rented', 'owned', or 'norent_noown'."),
    ("hopefully_in_the_future", "Out-of-range value error: house ownership must be 'rented', 'owned', or 'norent_noown'."),
    ("owned", None),
    ("rented", None),
    ("norent_noown", None)
])
def test_error_message_for_house_ownership_out_of_range(valid_inputs, house_ownership_value, expected_error_message):
    inputs = valid_inputs.copy()
    inputs["house_ownership"] = house_ownership_value
    assert check_out_of_range_values(inputs) == expected_error_message


# Current house years out-of-range
@pytest.mark.parametrize("current_house_yrs_value, expected_error_message", [
    (-50, "Out-of-range value error: current house years must be 10-14."), 
    (0, "Out-of-range value error: current house years must be 10-14."), 
    (9, "Out-of-range value error: current house years must be 10-14."), 
    (10, None), 
    (14, None), 
    (15, "Out-of-range value error: current house years must be 10-14."), 
    (1000, "Out-of-range value error: current house years must be 10-14.")
])
def test_error_message_for_current_house_yrs_out_of_range(valid_inputs, current_house_yrs_value, expected_error_message):
    inputs = valid_inputs.copy()
    inputs["current_house_yrs"] = current_house_yrs_value
    assert check_out_of_range_values(inputs) == expected_error_message


# Experience out-of-range
@pytest.mark.parametrize("experience_value, expected_error_message", [
    (-50, "Out-of-range value error: experience must be 0-20 years."), 
    (-1, "Out-of-range value error: experience must be 0-20 years."), 
    (0, None), 
    (20, None), 
    (21, "Out-of-range value error: experience must be 0-20 years."), 
    (1000, "Out-of-range value error: experience must be 0-20 years.")
])
def test_error_message_for_experience_out_of_range(valid_inputs, experience_value, expected_error_message):
    inputs = valid_inputs.copy()
    inputs["experience"] = experience_value
    assert check_out_of_range_values(inputs) == expected_error_message


# Current job years out-of-range
@pytest.mark.parametrize("current_job_yrs_value, expected_error_message", [
    (-50, "Out-of-range value error: current job years must be 0-14."), 
    (-1, "Out-of-range value error: current job years must be 0-14."), 
    (0, None), 
    (14, None), 
    (15, "Out-of-range value error: current job years must be 0-14."), 
    (1000, "Out-of-range value error: current job years must be 0-14.")
])
def test_error_message_for_current_job_yrs_out_of_range(valid_inputs, current_job_yrs_value, expected_error_message):
    inputs = valid_inputs.copy()
    inputs["current_job_yrs"] = current_job_yrs_value
    assert check_out_of_range_values(inputs) == expected_error_message
