# --- Imports ---
# Standard library imports
import warnings
from unittest.mock import patch

# Third-party library imports
import pytest
from fastapi.testclient import TestClient

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Local imports
from frontend.app import predict_loan_default
from backend.app import app

# --- Function .predict_loan_default() ---
class TestPredictLoanDefault:
    # Happy path
    @pytest.mark.integration
    @patch("frontend.app.requests.post")
    def test_happy_path(self, mock_post_request):
        # Raw inputs from Gradio UI as dictionary
        inputs = {
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
        }
        # Mock the post request to redirect it to the FastAPI test client
        client = TestClient(app)        
        def redirect_post_request_to_testclient(url, json, timeout):
            return client.post("/predict", json=json)
        mock_post_request.side_effect = redirect_post_request_to_testclient

        # Call .predict_loan_default()
        prediction, probabilities = predict_loan_default(**inputs)

        # Ensure requests.post() was called once
        mock_post_request.assert_called_once()
        # Ensure prediction is as expected
        assert isinstance(prediction, str)
        assert prediction in ["Default", "No Default"]
        # Ensure probabilities is as expected
        assert isinstance(probabilities, dict)
        assert "Default" in probabilities
        assert "No Default" in probabilities
        assert isinstance(probabilities["Default"], float)
        assert isinstance(probabilities["No Default"], float)
        # Ensure probabilities sum to approximately 1
        assert (probabilities["Default"] + probabilities["No Default"]) == pytest.approx(1.0)  # default relative tolerance of 1e-6 (0.0001%) and absolute tolerance of 1e-12
