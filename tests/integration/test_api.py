# --- Imports ---
# Standard library imports
import os

# Third-party library imports
import pytest 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Local imports
from api.app import load_pipeline


# --- Function .load_pipeline() ---
class TestLoadPipeline:
    @pytest.mark.integration
    def test_happy_path_with_minimal_pipeline(self, tmp_path):
        # Create minimal pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf_classifier", RandomForestClassifier())
        ])

        # Train with minimal data
        X = [[0, 0], [1, 1]]
        y = [0, 1]
        pipeline.fit(X, y)

        # Save to temporary file
        pipeline_path = tmp_path / "example_pipeline.joblib"
        joblib.dump(pipeline, pipeline_path)

        # Call .load_pipeline() function
        loaded_pipeline = load_pipeline(pipeline_path)

        # Call .pred_proba() method (with single sample)
        predicted_probabilities = loaded_pipeline.predict_proba([[0, 0]])

        # Ensure pipeline is scikit-learn Pipeline class with "predict_proba" method
        assert isinstance(loaded_pipeline, Pipeline)
        assert hasattr(loaded_pipeline, "predict_proba")
        # Ensure .predict_proba() returns numpy 2darray with 1 row and two columns (for classes 0 and 1)
        assert predicted_probabilities.shape == (1, 2)

    @pytest.mark.integration
    def test_happy_path_with_real_pipeline(self):
        # Create the pipeline path
        pipeline_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),  # path to directory of current file
            "../..",  # two levels up to root directory 
            "models", 
            "loan_default_rf_pipeline.joblib"
        )        
        # Call .load_pipeline() function
        loaded_pipeline = load_pipeline(pipeline_path)
        # Ensure pipeline is scikit-learn Pipeline class with "predict_proba" method
        assert isinstance(loaded_pipeline, Pipeline)
        assert hasattr(loaded_pipeline, "predict_proba")
