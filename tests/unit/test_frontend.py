# Standard library imports
import warnings

# Third-party library imports
import pytest

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  

# Local imports
from frontend.app import (
    snake_case_format, 
    snake_case_format_inputs, 
    format_house_ownership
)


# Define valid input dictionary for testing 
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


# --- snake_case_format() ---
# Ensure snake_case_format() converts strings to snake_case correctly
@pytest.mark.unit
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
def test_snake_case_format_happy_path(input_value, expected_output):
    assert snake_case_format(input_value) == expected_output


# --- snake_case_format_inputs() ---
# Ensure snake_case_format_inputs() formats all string values in a dictionary in snake_case
@pytest.mark.unit
def test_snake_case_format_inputs_happy_path():
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
    assert snake_case_format_inputs(raw_inputs) == expected_standardized_inputs

# Ensure snake_case_format_inputs() leaves inputs that are already formatted in snake_case unchanged 
@pytest.mark.unit
def test_snake_case_formatted_inputs_remain_unchanged():
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
    assert snake_case_format_inputs(standardized_inputs) == standardized_inputs


# --- format_house_ownership() ---
# Ensure format_house_ownership() converts display label "neither_rented_nor_owned" to label "norent_noown" as pipeline expects
@pytest.mark.unit
@pytest.mark.parametrize("display_label, expected_pipeline_label", [
    ("neither_rented_nor_owned", "norent_noown"),
    ("rented", "rented"),
    ("owned", "owned")
])
def test_format_house_ownership_happy_path(display_label, expected_pipeline_label):
    assert format_house_ownership(display_label) == expected_pipeline_label
