# --- Imports ---
# Standard library imports
from unittest.mock import patch, MagicMock
from pathlib import Path

# Third-party library imports
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

# Local imports
from api.app import load_pipeline, app


# --- Function .load_pipeline() ---
class TestLoadPipeline:
    @patch("api.app.joblib.load")
    @patch("api.app.Path.exists")
    def test_happy_path_with_mock_pipeline(self, mock_path_exists, mock_joblib_load):
        # Simulate that the file exists
        mock_path_exists.return_value = True
        # Simulate loaded pipeline instance
        mock_pipeline = MagicMock(spec=Pipeline)
        mock_pipeline.predict_proba = MagicMock()
        mock_joblib_load.return_value = mock_pipeline

        # Call .load_pipeline()
        pipeline = load_pipeline("some_path.joblib")

        # Ensure loaded pipeline is a mock pipeline with a "predict_proba" attribute
        assert pipeline is mock_pipeline
        assert hasattr(pipeline, "predict_proba")
        # Ensure Path.exists was called once
        mock_path_exists.assert_called_once()
        # Ensure joblib.load was called once
        mock_joblib_load.assert_called_once_with("some_path.joblib")

    @patch("api.app.joblib.load")
    @patch("api.app.Path.exists")
    def test_accepts_pathlike_object(self, mock_path_exists, mock_joblib_load):
        # Simulate that the file exists
        mock_path_exists.return_value = True
        # Simulate loaded pipeline instance
        mock_pipeline = MagicMock(spec=Pipeline)
        mock_pipeline.predict_proba = MagicMock()
        mock_joblib_load.return_value = mock_pipeline

        # Call .load_pipeline() with a pathlib.Path object (path-like object)
        path_like_object = Path("some_path.joblib")
        pipeline = load_pipeline(path_like_object)

        # Ensure loaded pipeline is a mock pipeline with a "predict_proba" attribute
        assert pipeline is mock_pipeline
        assert hasattr(pipeline, "predict_proba")
        # Ensure Path.exists was called once
        mock_path_exists.assert_called_once()
        # Ensure joblib.load was called once with converted string 
        mock_joblib_load.assert_called_once_with(str(path_like_object))

    @patch("api.app.Path.exists")
    def test_raises_file_not_found_error_for_non_existent_file(self, mock_path_exists):
        # Simulate that the file does not exist
        mock_path_exists.return_value = False
        # Ensure FileNotFoundError is raised
        with pytest.raises(FileNotFoundError) as exc_info:
            load_pipeline("non_existent_file.joblib")
        # Ensure error message is as expected
        error_msg = str(exc_info.value)
        assert "Pipeline file not found at" in error_msg
        assert "non_existent_file.joblib" in error_msg
        # Ensure Path.exists was called 
        mock_path_exists.assert_called_once()

    @patch("api.app.joblib.load")
    @patch("api.app.Path.exists")
    def test_raises_runtime_error_if_joblib_load_fails(self, mock_path_exists, mock_joblib_load):
        # Simulate that the file exists
        mock_path_exists.return_value = True
        # Simulate an error when loading joblib file
        mock_joblib_load.side_effect = Exception("joblib load error")

        # Ensure RuntimeError is raised
        with pytest.raises(RuntimeError) as exc_info:
            load_pipeline("some_path.joblib")
        # Ensure error message is as expected
        error_msg = str(exc_info.value)
        assert "Failed to load pipeline" in error_msg
        assert "some_path.joblib" in error_msg
        # Ensure the original error was propagated
        propagated_error = exc_info.value.__cause__
        assert isinstance(propagated_error, Exception)
        assert str(propagated_error) == "joblib load error"
        # Ensure Path.exists() and joblib.load() were called
        mock_path_exists.assert_called_once()
        mock_joblib_load.assert_called_once_with("some_path.joblib")

    @patch("api.app.joblib.load")
    @patch("api.app.Path.exists")
    def test_raises_type_error_if_loaded_object_is_not_pipeline(self, mock_path_exists, mock_joblib_load):
        # Simulate that pipeline file exists
        mock_path_exists.return_value = True
        # Simulate loaded object that is not a pipeline
        mock_joblib_load.return_value = "not a pipeline"

        # Ensure .load_pipeline() raises TypeError
        with pytest.raises(TypeError) as exc_info:
            load_pipeline("some_path.joblib")
        # Ensure error message is as expected
        error_msg = str(exc_info.value)
        assert "Loaded object is not a scikit-learn Pipeline" in error_msg
        # Ensure Path.exists() and joblib.load() were called
        mock_path_exists.assert_called_once()
        mock_joblib_load.assert_called_once_with("some_path.joblib")

    @patch("api.app.joblib.load")
    @patch("api.app.Path.exists")
    def test_raises_type_error_if_predict_proba_does_not_exist(self, mock_path_exists, mock_joblib_load):
        # Simulate that pipeline file exists
        mock_path_exists.return_value = True 
        # Simulate loaded pipeline instance without a "predict_proba" method
        mock_pipeline = MagicMock(spec=Pipeline)
        del mock_pipeline.predict_proba
        mock_joblib_load.return_value = mock_pipeline

        # Ensure .load_pipeline() raises TypeError
        with pytest.raises(TypeError) as exc_info:
            load_pipeline("some_path.joblib")
        # Ensure error message is as expected
        error_msg = str(exc_info.value)
        assert "Loaded pipeline does not have a .predict_proba() method" in error_msg
        # Ensure Path.exists() and joblib.load() were called
        mock_path_exists.assert_called_once()
        mock_joblib_load.assert_called_once_with("some_path.joblib")


# --- FastAPI endpoint "/predict" ---
client = TestClient(app)

class TestPredict:
    @patch("api.app.pipeline.predict_proba")
    def test_happy_path_single_input(self, mock_predict_proba):
        valid_single_input = {
            "income": 300000,
            "age": 30,
            "experience": 3,
            "married": "single",
            "house_ownership": "rented",
            "car_ownership": "no",
            "profession": "artist",
            "city": "sikar",
            "state": "rajasthan",
            "current_job_yrs": 3,
            "current_house_yrs": 11           
        }
        mock_predict_proba.return_value = np.array([[0.8, 0.2]])
        
        response = client.post("/predict", json=valid_single_input)

        # Ensure post request was successful
        assert response.status_code == 200
        # Ensure prediction response is as expected
        prediction_response = response.json()
        expected_prediction_response = {
            "results": [{
                "prediction": "No Default",  # 0.2 < 0.29 threshold
                "probabilities": {
                    "Default": 0.2, 
                    "No Default": 0.8
                }
            }],
            "n_predictions": 1
        }
        assert prediction_response == expected_prediction_response

    @patch("api.app.pipeline.predict_proba")
    def test_happy_path_batch_input(self, mock_predict_proba):
        valid_batch_input = [
            {
                "income": 300000,
                "age": 30,
                "experience": 3,
                "married": "single",
                "house_ownership": "rented",
                "car_ownership": "no",
                "profession": "artist",
                "city": "sikar",
                "state": "rajasthan",
                "current_job_yrs": 3,
                "current_house_yrs": 11           
            },
            {
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
        ]
        mock_predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        
        response = client.post("/predict", json=valid_batch_input)

        # Ensure post request was successful
        assert response.status_code == 200
        # Ensure prediction response is as expected
        prediction_response = response.json()
        expected_prediction_response = {
            "results": [
                {
                    "prediction": "No Default",  # 0.2 < 0.29 threshold
                    "probabilities": {
                        "Default": 0.2, 
                        "No Default": 0.8
                    }
                },
                {
                    "prediction": "Default",  
                    "probabilities": {
                        "Default": 0.7, 
                        "No Default": 0.3
                    }
                }
            ],
            "n_predictions": 2
        }
        assert prediction_response == expected_prediction_response

    @patch("api.app.pipeline.predict_proba")
    def test_happy_path_input_standardization(self, mock_predict_proba):
        valid_single_input = {
            "income": 300000,
            "age": 30,
            "experience": 3,
            "married": "single",
            "house_ownership": "rented",
            "car_ownership": "no",
            "profession": "artist",
            "city": "sikar",
            "state": "rajasthan",
            "current_job_yrs": 3,
            "current_house_yrs": 11           
        }
        mock_predict_proba.return_value = np.array([[0.8, 0.2]])
        expected_df = pd.DataFrame([valid_single_input])
        
        # Make post request to predict endpoint with test client
        client.post("/predict", json=valid_single_input)

        # Ensure .predict_proba() was called once
        mock_predict_proba.assert_called_once()
        # Ensure .predict_proba() used the expected DataFrame
    