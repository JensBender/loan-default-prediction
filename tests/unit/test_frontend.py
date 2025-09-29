# Standard library imports
import warnings
import logging

# Third-party library imports
import pytest

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  

# Local imports
from frontend.app import (
    format_snake_case, 
    format_snake_case_in_dict, 
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

# --- Function .format_snake_case_in_dict() ---
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
        assert format_snake_case_in_dict(inputs) == expected_outputs

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
        assert format_snake_case_in_dict(inputs_with_snake_case) == inputs_with_snake_case


# --- Function .format_house_ownership() ---
class TestFormatHouseOwnership:
    # Happy path
    @pytest.mark.unit
    @pytest.mark.parametrize("display_label, expected_pipeline_label", [
        ("neither_rented_nor_owned", "norent_noown"),
        ("rented", "rented"),
        ("owned", "owned")
    ])
    def test_happy_path(self, display_label, expected_pipeline_label):
        assert format_house_ownership(display_label) == expected_pipeline_label
    
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
        assert format_house_ownership(non_string_value) == non_string_value 


# --- Function ._format_validation_error() ---
class TestFormatValidationError:
    # Single field happy path
    @pytest.mark.unit
    @pytest.mark.parametrize("field, partial_error_msg", [
        ("age", "Age: Enter a number between 21 and 79."),
        ("married", "Married/Single: Select 'Married' or 'Single'"),
        ("income", "Income: Enter a number that is 0 or greater."),
        ("car_ownership", "Car Ownership: Select 'Yes' or 'No'."),
        ("house_ownership", "House Ownership: Select 'Rented', 'Owned', or 'Neither Rented Nor Owned'."),
        ("current_house_yrs", "Current House Years: Enter a number between 10 and 14."),
        ("city", "City: Select a city from the list."),
        ("state", "State: Select a state from the list."),
        ("profession", "Profession: Select a profession from the list."),
        ("experience", "Experience: Enter a number between 0 and 20."),
        ("current_job_yrs", "Current Job Years: Enter a number between 0 and 14."),
    ])
    def test_single_field_happy_path(self, field, partial_error_msg):
        error_detail = {
            "detail": [{
                "type": "some error type",
                "loc": ["body", "PipelineInput", field],
                "msg": "some error message",
                "input": "some invalid input"
            }]
        }
        expected_error_msg = f"Input Error! Please check your inputs and try again.\n{partial_error_msg}\n"
        assert _format_validation_error(error_detail) == expected_error_msg

    # Multiple fields happy path
    @pytest.mark.unit
    @pytest.mark.parametrize("field_1, field_2, partial_error_message", [
        ("age", "married", "Age: Enter a number between 21 and 79.\nMarried/Single: Select 'Married' or 'Single'\n"),
        ("income", "car_ownership", "Income: Enter a number that is 0 or greater.\nCar Ownership: Select 'Yes' or 'No'.\n"),
        ("house_ownership", "current_house_yrs", "House Ownership: Select 'Rented', 'Owned', or 'Neither Rented Nor Owned'.\nCurrent House Years: Enter a number between 10 and 14.\n"),
        ("city", "state", "City: Select a city from the list.\nState: Select a state from the list.\n"),
        ("profession", "experience", "Profession: Select a profession from the list.\nExperience: Enter a number between 0 and 20.\n"),
        ("age", "current_job_yrs", "Age: Enter a number between 21 and 79.\nCurrent Job Years: Enter a number between 0 and 14.\n")
    ])
    def test_multiple_fields_happy_path(self, field_1, field_2, partial_error_message):
        error_detail = {
            "detail": [
                {
                    "type": "some error type",
                    "loc": ["body", "PipelineInput", field_1],
                    "msg": "some error message",
                    "input": "some invalid input"
                },
                {
                    "type": "some error type",
                    "loc": ["body", "PipelineInput", field_2],
                    "msg": "some error message",
                    "input": "some invalid input"
                }
            ]
        }
        expected_error_msg = f"Input Error! Please check your inputs and try again.\n{partial_error_message}"
        assert _format_validation_error(error_detail) == expected_error_msg
    
    # Empty error detail list
    @pytest.mark.unit
    def test_empty_error_detail_list(self, caplog):
        error_detail = {"detail": []}
        expected_error_msg = "Input Error! Please check your inputs and try again.\n"
        # Ensure error message is as expected
        assert _format_validation_error(error_detail) == expected_error_msg
        # Ensure no error was logged
        assert caplog.text == ""

    # All fields missing in error location
    @pytest.mark.unit
    def test_all_fields_missing_in_error_location(self, caplog):
        error_detail = {
            "detail": [{
                "type": "some error type",
                "loc": ["body", "PipelineInput", "some_field"],  # all input fields missing 
                "msg": "some error message",
                "input": "some invalid input"
            }]
        }
        expected_error_msg = "Input Error! Please check your inputs and try again.\n"
        # Ensure error message is as expected
        assert _format_validation_error(error_detail) == expected_error_msg
        # Ensure no error was logged
        assert caplog.text == ""

    # Unexpected Pydantic error format
    @pytest.mark.unit
    @pytest.mark.parametrize("unexpected_error_format", [
        None,  
        {},  # "detail" key missing
        {"detail": "a string"},  # "detail" value not a list
        {"detail": ["a string"]},  # "detail" list element not a dictionary
        {"detail": [{}]},  # no "loc" key
        {"detail": [{"loc": 123}]}  # "loc" value not a list 
    ])
    def test_unexpected_error_format(self, unexpected_error_format, caplog):
        expected_error_msg = "Input Error! Please check your inputs and try again.\n"
        with caplog.at_level(logging.WARNING):
            error_msg = _format_validation_error(unexpected_error_format)
            # Ensure error message is as expected
            assert error_msg == expected_error_msg
            # Ensure warning was logged
            assert len(caplog.records) == 1
            log_record = caplog.records[0]
            assert log_record.levelname == "WARNING"
            assert "Failed to parse validation error from backend" in log_record.message


# --- Function .predict_loan_default() ---