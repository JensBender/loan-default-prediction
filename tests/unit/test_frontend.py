# Standard library imports
import warnings
import logging
from unittest.mock import patch, MagicMock

# Third-party library imports
import pytest
import requests

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
        # Ensure no logged messages
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
        # Ensure no logged messages
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
class TestPredictLoanDefault:
    # Input preprocessing happy path
    @pytest.mark.unit
    @pytest.mark.parametrize("raw_inputs, expected_json", [
        # Test case 1
        (
            # Raw inputs from Gradio UI
            {
                "age": 30, 
                "married": "Single", 
                "income": 300000, 
                "car_ownership": "No", 
                "house_ownership": "Neither Rented Nor Owned", 
                "current_house_yrs": 11, 
                "city": "Sikar", 
                "state": "Rajasthan", 
                "profession": "Artist", 
                "experience": 3, 
                "current_job_yrs": 3
            },
            # Expected JSON body of post request 
            {
                "age": 30,
                "married": "single",  # snake_case
                "income": 300000,
                "car_ownership": "no",  # snake_case
                "house_ownership": "norent_noown",  # snake_case + special formatting
                "current_house_yrs": 11,
                "city": "sikar",  # snake_case
                "state": "rajasthan",  # snake_case
                "profession": "artist",  # snake_case
                "experience": 3,
                "current_job_yrs": 3
            }
        ),
        # Test case 2
        (
            # Raw inputs from Gradio UI
            {
                "age": 45, 
                "married": "Married", 
                "income": 500000, 
                "car_ownership": "Yes", 
                "house_ownership": "Rented", 
                "current_house_yrs": 12, 
                "city": "Bhalswa Jahangir Pur", 
                "state": "Jammu And Kashmir", 
                "profession": "Air Traffic Controller", 
                "experience": 15, 
                "current_job_yrs": 5
            },
            # Expected JSON body of post request 
            {
                "age": 45,
                "married": "married",  # snake_case
                "income": 500000,
                "car_ownership": "yes",  # snake_case
                "house_ownership": "rented",  # snake_case + special formatting
                "current_house_yrs": 12,
                "city": "bhalswa_jahangir_pur",  # snake_case
                "state": "jammu_and_kashmir",  # snake_case
                "profession": "air_traffic_controller",  # snake_case
                "experience": 15,
                "current_job_yrs": 5
            }
        ),
        # Test case 3
        (
            # Raw inputs from Gradio UI
            {
                "age": 60, 
                "married": "Single", 
                "income": 100000, 
                "car_ownership": "No", 
                "house_ownership": "Owned", 
                "current_house_yrs": 4, 
                "city": "Mira Bhayandar", 
                "state": "Uttar Pradesh", 
                "profession": "Software Developer", 
                "experience": 20, 
                "current_job_yrs": 10
            },
            # Expected JSON body of post request 
            {
                "age": 60,
                "married": "single",  # snake_case
                "income": 100000,
                "car_ownership": "no",  # snake_case
                "house_ownership": "owned",  # snake_case + special formatting
                "current_house_yrs": 4,
                "city": "mira_bhayandar",  # snake_case
                "state": "uttar_pradesh",  # snake_case
                "profession": "software_developer",  # snake_case
                "experience": 20,
                "current_job_yrs": 10
            }
        )
    ])
    @patch("frontend.app.requests.post")
    def test_input_preprocessing_happy_path(self, mock_post_request, raw_inputs, expected_json):
        # Simulate the post request 
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{
                "prediction": "No Default",  
                "probabilities": {
                    "Default": 0.2, 
                    "No Default": 0.8
                }
            }],
            "n_predictions": 1
        }
        mock_post_request.return_value = mock_response

        # Call .predict_loan_default() 
        predict_loan_default(**raw_inputs)

        # Ensure requests.post() was called once
        mock_post_request.assert_called_once()
        # Get positional and keyword arguments used in the call
        args, kwargs = mock_post_request.call_args
        # Extract the JSON body from the keyword call arguments
        json = kwargs["json"]
        # Ensure requests.post() was called with the expected json body
        assert json == expected_json

    # Response parsing happy path
    @pytest.mark.unit
    @patch("frontend.app.requests.post")
    def test_response_parsing_happy_path(self, mock_post_request):
        # Simulate the post request 
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{
                "prediction": "No Default",  
                "probabilities": {
                    "Default": 0.2, 
                    "No Default": 0.8
                }
            }],
            "n_predictions": 1
        }
        mock_post_request.return_value = mock_response
        # Expected prediction result
        expected_prediction = "No Default"
        expected_probabilities = {
            "Default": 0.2, 
            "No Default": 0.8
        }

        # Call .predict_loan_default()
        prediction, probabilities = predict_loan_default(
            age=30,
            married="Single",
            income=300000,
            car_ownership="No",
            house_ownership="Neither Rented Nor Owned",
            current_house_yrs=11,
            city="Sikar",
            state="Rajasthan",
            profession="Artist",
            experience=3,
            current_job_yrs=3
        )

        # Ensure requests.post() was called once
        mock_post_request.assert_called_once()
        # Ensure prediction and probabilities are as expected
        prediction == expected_prediction
        probabilities == expected_probabilities

    # Response parsing error 
    # KeyError for "results", "prediction" or "probabilities"
    # IndexError for results[0] 
    # TypeError for prediction_response["results"] = Not a list (or iterable) or results list element not a dictionary
    @pytest.mark.unit
    @patch("frontend.app.requests.post")
    def test_response_parsing_error(self, mock_post_request, caplog):
        # Simulate the post request 
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {}  # "results" key missing
        mock_post_request.return_value = mock_response

        # Call .predict_loan_default() and capture error logs
        with caplog.at_level(logging.ERROR):
            prediction, probabilities = predict_loan_default(
                age=30,
                married="Single",
                income=300000,
                car_ownership="No",
                house_ownership="Neither Rented Nor Owned",
                current_house_yrs=11,
                city="Sikar",
                state="Rajasthan",
                profession="Artist",
                experience=3,
                current_job_yrs=3
            )

        # Ensure requests.post() was called once
        mock_post_request.assert_called_once()
        # Ensure expected error messages for Gradio frontend
        assert prediction == "Prediction Response Error"
        assert probabilities == "The prediction service returned an invalid prediction format."
        # Ensure exactly one error was logged with correct level and message
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "ERROR"
        assert "Failed to parse prediction response from backend." in caplog.records[0].message

    # Handle HTTP 422 pydantic validation errors
    # Handle other HTTP errors
    # Handle ConnectionError
    # Handle Timeout
    # Handle RequestException