# Standard library imports
import warnings

# Third-party library imports
import pytest

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  

# Local imports
from frontend.app import (
    format_snake_case, 
    snake_case_str_values_in_dict, 
    format_house_ownership,
    _format_validation_error,
    predict_loan_default
)


# --- Function .format_snake_case() ---
class TestFormatSnakeCase:
    # Happy path
    @pytest.mark.unit
    @pytest.mark.parametrize("input, expected_output", [
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
    def test_happy_path(self, input, expected_output):
        assert format_snake_case(input) == expected_output
    
    # Non-string values remain unchanged
    @pytest.mark.unit
    @pytest.mark.parametrize("non_string_value", [
        1,
        1.23,
        False,
        None,  
        ["a", "list"],
        ("a", "tuple"),
        {"a": "dictionary"},
        {"a", "set"}        
    ])
    def test_non_string_values_remain_unchanged(self, non_string_value):
        assert format_snake_case(non_string_value) == non_string_value

# --- Function .snake_case_str_values_in_dict() ---
class TestSnakeCaseInDict:
    # Happy path
    @pytest.mark.unit
    def test_happy_path(self):
        inputs = {
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
        expected_outputs = {
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
        assert snake_case_str_values_in_dict(inputs) == expected_outputs

    # Inputs that are already in snake_case remain unchanged 
    @pytest.mark.unit
    def test_snake_case_formatted_inputs_remain_unchanged(self):
        inputs_with_snake_case = {
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
        assert snake_case_str_values_in_dict(inputs_with_snake_case) == inputs_with_snake_case


# --- Function .format_house_ownership() ---
# Happy path
@pytest.mark.parametrize("display_label, expected_pipeline_label", [
    ("neither_rented_nor_owned", "norent_noown"),
    ("rented", "rented"),
    ("owned", "owned")
])
def test_format_house_ownership_happy_path(display_label, expected_pipeline_label):
    assert format_house_ownership(display_label) == expected_pipeline_label

# --- Function ._format_validation_error() ---
# --- Function .predict_loan_default() ---