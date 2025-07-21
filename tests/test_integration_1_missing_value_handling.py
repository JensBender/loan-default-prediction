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


# --- Pipeline Segment to Be Integration Tested ---
@pytest.fixture
def pipeline_segment(): 
    return Pipeline([
        ("missing_value_checker", MissingValueChecker(critical_features=CRITICAL_FEATURES, non_critical_features=NON_CRITICAL_FEATURES)),
        ("missing_value_handler", ColumnTransformer(
            transformers=[("categorical_imputer", SimpleImputer(strategy="most_frequent").set_output(transform="pandas"), NON_CRITICAL_FEATURES)],
            remainder="passthrough",
            verbose_feature_names_out=False  # preserve input column names instead of adding prefix 
        ).set_output(transform="pandas")),  # output pd.DataFrame instead of np.array 
    ])


# --- Test Cases ---
# Ensure Pipeline .fit() raises MissingValueError for missing values in critical features
@pytest.mark.integration
@pytest.mark.parametrize("missing_value", [None, np.nan])
@pytest.mark.parametrize("critical_feature", CRITICAL_FEATURES)
def test_pipeline_fit_raises_missing_value_error_for_critical_features(X_input, pipeline_segment, missing_value, critical_feature):
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
        with pytest.raises(MissingValueError, match=expected_error_message):
            pipeline_segment.fit(X_with_missing_values)
