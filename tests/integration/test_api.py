# --- Imports ---
# Standard library imports
from pathlib import Path 

# Third-party library imports
import pytest 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
from fastapi.testclient import TestClient

# Local imports
from api.app import load_pipeline, app


# --- Function .load_pipeline() ---
class TestLoadPipeline:
    @pytest.mark.integration
    @pytest.mark.parametrize("path_input_type", ["str", "Path"])
    def test_happy_path_with_minimal_pipeline(self, path_input_type, tmp_path):
        # Create minimal pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf_classifier", RandomForestClassifier())
        ])
        # Train with minimal data
        X = [[0, 0], [1, 1]]
        y = [0, 1]
        pipeline.fit(X, y)
        # Create path to pipeline as both Path object and string
        pipeline_path: Path = tmp_path / "example_pipeline.joblib"
        pipeline_path_str: str = str(pipeline_path)

        # Save to temporary file
        joblib.dump(pipeline, pipeline_path_str)

        # Call .load_pipeline() function
        if path_input_type == "str":
            loaded_pipeline = load_pipeline(pipeline_path_str)
        else:  # path_input_type == "Path"
            loaded_pipeline = load_pipeline(pipeline_path)

        # Call .pred_proba() method (with single sample)
        predicted_probabilities = loaded_pipeline.predict_proba([[0, 0]])

        # Ensure pipeline is scikit-learn Pipeline class with "predict_proba" method
        assert isinstance(loaded_pipeline, Pipeline)
        assert hasattr(loaded_pipeline, "predict_proba")
        # Ensure .predict_proba() returns numpy 2darray with 1 row and two columns (for classes 0 and 1)
        assert predicted_probabilities.shape == (1, 2)

    @pytest.mark.integration
    @pytest.mark.parametrize("path_input_type", ["str", "Path"])
    def test_happy_path_with_real_pipeline(self, path_input_type, request):
        # Get LocalPath to root directory using pytest's built-in request fixture
        root_dir = request.config.rootdir
        # Convert LocalPath to Path object 
        root_dir = Path(str(root_dir))
        # Get path to pipeline 
        pipeline_path = root_dir / "models" / "loan_default_rf_pipeline.joblib"  
        # Valid pipeline input (single-row DataFrame)
        valid_input = pd.DataFrame({
            "income": [300000],
            "age": [30],
            "experience": [3],
            "married": ["single"],
            "house_ownership": ["rented"],
            "car_ownership": ["no"],
            "profession": ["Artist"],
            "city": ["Sikar"],
            "state": ["Rajasthan"],
            "current_job_yrs": [3],
            "current_house_yrs": [11],
        })  

        # Call .load_pipeline() function
        if path_input_type == "str":
            pipeline_path = str(pipeline_path)
        loaded_pipeline = load_pipeline(pipeline_path)
        # Predict probabilities with valid input
        predicted_probabilities = loaded_pipeline.predict_proba(valid_input)

        # Ensure pipeline is scikit-learn Pipeline class with "predict_proba" method
        assert isinstance(loaded_pipeline, Pipeline)
        assert hasattr(loaded_pipeline, "predict_proba")
        # Ensure .pred_proba() returns numpy array with 1 row and 2 columns (for the 2 classes)
        assert isinstance(predicted_probabilities, np.ndarray)
        assert predicted_probabilities.shape == (1, 2)

    @pytest.mark.integration
    @pytest.mark.parametrize("invalid_path_type", [
        None,  
        1,
        1.23,
        False,
        ["a", "list"],
        ("a", "tuple"),
        {"a": "dictionary"},
        {"a", "set"}
    ])
    def test_raises_type_error_for_invalid_path_type(self, invalid_path_type):
        # Ensure .load_pipeline() raises TypeError
        with pytest.raises(TypeError) as exc_info:
            load_pipeline(invalid_path_type)
        # Ensure error message is as expected
        error_msg = str(exc_info.value)
        assert "'path' must be a string or Path object" in error_msg   

    @pytest.mark.integration
    def test_raises_file_not_found_error_for_non_existent_file(self):
        # Ensure .load_pipeline() raises FileNotFoundError
        with pytest.raises(FileNotFoundError) as exc_info:
            load_pipeline("non_existent_file.joblib")
        # Ensure error message is as expected
        error_msg = str(exc_info.value)
        assert "Pipeline file not found at" in error_msg
        assert "non_existent_file.joblib" in error_msg

    @pytest.mark.integration
    def test_raises_runtime_error_if_joblib_load_fails(self, tmp_path):
        # Create a corrupt (non-joblib) file
        corrupt_file_path = tmp_path / "corrupt.joblib"
        with open(corrupt_file_path, "w") as file:
            file.write("This is not a joblib file")

        # Ensure .load_pipeline() raises RuntimeError 
        with pytest.raises(RuntimeError) as exc_info:
            load_pipeline(corrupt_file_path)
        # Ensure error message is as expected
        error_msg = str(exc_info.value)
        assert "Failed to load pipeline" in error_msg
        assert "corrupt.joblib" in error_msg

    @pytest.mark.integration
    def test_raises_type_error_if_loaded_object_is_not_pipeline(self, tmp_path):
        # Create a joblib file that is not a Pipeline
        not_a_pipeline = {"a": "dictionary"}
        file_path = tmp_path / "not_a_pipeline.joblib"
        joblib.dump(not_a_pipeline, file_path)

        # Ensure .load_pipeline() raises TypeError
        with pytest.raises(TypeError) as exc_info:
            load_pipeline(file_path)
        # Ensure error message is as expected
        error_msg = str(exc_info.value)
        assert "Loaded object is not a scikit-learn Pipeline" in error_msg    

    @pytest.mark.integration
    def test_raises_type_error_if_predict_proba_does_not_exist(self, tmp_path):
        # Create a pipeline without .predict_proba() method (Regressor instead of Classifier)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf_regressor", RandomForestRegressor())
        ])
        # Save pipeline to temporary file
        pipeline_path = tmp_path / "pipeline.joblib"
        joblib.dump(pipeline, pipeline_path)

        # Ensure .load_pipeline() raises TypeError
        with pytest.raises(TypeError) as exc_info:
            load_pipeline(pipeline_path)
        # Ensure error message is as expected
        error_msg = str(exc_info.value)
        assert "Loaded pipeline does not have a .predict_proba() method" in error_msg


# --- FastAPI endpoint "/predict" ---
client = TestClient(app)

class TestPredict:
    @pytest.mark.integration
    def test_single_input_happy_path(self):
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

        # Post request to predict endpoint
        response = client.post("/predict", json=valid_single_input)

        # Ensure post request was successful
        assert response.status_code == 200
        # Ensure response contains expected "results" 
        prediction_response = response.json()
        assert "results" in prediction_response
        assert isinstance(prediction_response["results"], list)
        assert len(prediction_response["results"]) == 1
        assert isinstance(prediction_response["results"][0], dict)
        assert "prediction" in prediction_response["results"][0]
        assert "probabilities" in prediction_response["results"][0]
        assert isinstance(prediction_response["results"][0]["probabilities"], dict)
        assert "Default" in prediction_response["results"][0]["probabilities"]
        assert "No Default" in prediction_response["results"][0]["probabilities"]
        # Ensure response contains expected "n_predictions"
        assert "n_predictions" in prediction_response
        assert isinstance(prediction_response["n_predictions"], int)
        assert prediction_response["n_predictions"] == 1
