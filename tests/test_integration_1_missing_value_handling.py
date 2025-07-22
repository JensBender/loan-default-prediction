# Standard library imports
import os
import sys

# Third-party library imports
import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Add the parent directory to the path (for local imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local imports
from app.custom_transformers import MissingValueChecker, MissingValueError
from app.global_constants import CRITICAL_FEATURES, NON_CRITICAL_FEATURES


# --- Fixtures ---
# Fixture to create X input DataFrame for use in tests
@pytest.fixture
def X_input():
    return pd.DataFrame({
        "income": [9121364, 2636544, 9470213, 6558967, 6245331, 154867],
        "age": [70, 39, 41, 41, 65, 64],
        "experience": [18, 0, 5, 10, 6, 1],
        "married": ["single", "single", "single", "married", "single", "single"],
        "house_ownership": ["rented", "rented", "norent_noown", "rented", "rented", "owned"],
        "car_ownership": ["no", "no", "yes", "no", "no", "no"],
        "profession": ["Artist", "Computer_hardware_engineer", "Web_designer", "Comedian", 
                       "Financial_Analyst", "Statistician"],
        "city": ["Sikar", "Vellore", "Bidar", "Bongaigaon", "Eluru[25]", "Danapur"],
        "state": ["Rajasthan", "Tamil_Nadu", "Karnataka", "Assam", "Andhra_Pradesh", "Bihar"],
        "current_job_yrs": [3, 0, 5, 10, 6, 1],
        "current_house_yrs": [11, 11, 13, 12, 12, 12],
    })

# Fixture to create pipeline segment for use in tests
@pytest.fixture
def pipeline(): 
    return Pipeline([
        ("missing_value_checker", MissingValueChecker(critical_features=CRITICAL_FEATURES, non_critical_features=NON_CRITICAL_FEATURES)),
        ("missing_value_handler", ColumnTransformer(
            transformers=[("categorical_imputer", SimpleImputer(strategy="most_frequent").set_output(transform="pandas"), NON_CRITICAL_FEATURES)],
            remainder="passthrough",
            verbose_feature_names_out=False  # preserve input column names instead of adding prefix 
        ).set_output(transform="pandas")),  # output pd.DataFrame instead of np.array 
    ])


# --- Test Cases ---
# Ensure pipeline .fit() and .transform() raise MissingValueError for missing values in critical features
@pytest.mark.integration
@pytest.mark.parametrize("method", ["fit", "transform"])
@pytest.mark.parametrize("missing_value", [None, np.nan])
@pytest.mark.parametrize("critical_feature", CRITICAL_FEATURES)
def test_pipeline_fit_and_transform_raise_missing_value_error_for_critical_features(X_input, pipeline, method, missing_value, critical_feature):
        X = X_input.copy()
        X_with_missing_values = X_input.copy()
        X_with_missing_values[critical_feature] = missing_value
        # Create expected dictionary of number of missing values by column 
        expected_missing_by_column_dict = {"income": 0, "age": 0, "experience": 0, "profession": 0, "city": 0, "state": 0, "current_job_yrs": 0, "current_house_yrs": 0}
        expected_missing_by_column_dict[critical_feature] = 6  # X_input has 6 rows
        # Create expected error message text
        expected_error_message = (
            f"6 missing values found in critical features "
            f"across 6 rows. Please provide missing values.\n"
            f"Missing values by column: {expected_missing_by_column_dict}" 
        )
        # Ensure .fit() raises MissingValueError with expected error message text
        if method == "fit":
            with pytest.raises(MissingValueError, match=expected_error_message):
                pipeline.fit(X_with_missing_values)
        # Ensure .transform() raises MissingValueError with expected error message text
        else:
            # Fit on original DataFrame, but transform on DataFrame with missing values
            pipeline.fit(X)
            with pytest.raises(MissingValueError, match=expected_error_message):
                pipeline.transform(X_with_missing_values)

# Ensure pipline .fit() imputes missing values in non-critical features
@pytest.mark.integration
@pytest.mark.parametrize("missing_value", [None, np.nan])
@pytest.mark.parametrize("non_critical_feature", NON_CRITICAL_FEATURES)
def test_pipeline_fit_warns_and_learns_mode_for_missing_values_in_non_critical_features(X_input, pipeline, missing_value, non_critical_feature, capsys):
        X_with_missing_value = X_input.copy()
        X_with_missing_value.loc[0, non_critical_feature] = missing_value  # use first row as a representative example
        # Ensure .fit() prints warning message
        pipeline.fit(X_with_missing_value)
        captured_output_and_error = capsys.readouterr()
        warning_message = captured_output_and_error.out
        assert "Warning" in warning_message 
        assert "1 missing value found in non-critical features" in warning_message
        assert "will be imputed" in warning_message
        assert captured_output_and_error.err == ""
        # Ensure .fit() learns mode (most frequent value) of each non-critical feature
        X_transformed = pipeline.transform(X_with_missing_value)
        assert X_transformed.loc[0, "married"] == "single"
        assert X_transformed.loc[0, "house_ownership"] == "rented"
        assert X_transformed.loc[0, "car_ownership"] == "no"
        