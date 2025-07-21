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
pipeline_segment = Pipeline([
    ("missing_value_checker", MissingValueChecker(critical_features=CRITICAL_FEATURES, non_critical_features=NON_CRITICAL_FEATURES)),
    ("missing_value_handler", ColumnTransformer(
        transformers=[("categorical_imputer", SimpleImputer(strategy="most_frequent").set_output(transform="pandas"), NON_CRITICAL_FEATURES)],
        remainder="passthrough",
        verbose_feature_names_out=False  # preserve input column names instead of adding prefix 
    ).set_output(transform="pandas")),  # output pd.DataFrame instead of np.array 
])
