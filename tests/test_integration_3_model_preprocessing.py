# Standard library imports
import os
import sys

# Third-party library imports
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

# Add the parent directory to the path (for local imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local imports
from app.custom_transformers import FeatureSelector
from app.global_constants import NUMERICAL_COLUMNS, ORDINAL_COLUMN_ORDERS, COLUMNS_TO_KEEP
from tests.base_pipeline_tests import BasePipelineTests


# --- Fixtures ---
# Fixture to create X input DataFrame for use in tests
@pytest.fixture
def X_input():
    return pd.DataFrame({
        "income": [9121364, 2636544, 9470213, 6558967, 6245331, 154867],
        "age": [70, 39, 41, 41, 65, 64],
        "experience": [18, 0, 5, 10, 6, 1],
        "married": [False, False, False, True, False, False],
        "house_ownership": ["rented", "rented", "norent_noown", "rented", "rented", "owned"],
        "car_ownership": [False, False, True, False, False, False],
        "profession": ["artist", "computer_hardware_engineer", "web_designer", "comedian", 
                        "financial_analyst", "statistician"],
        "city": ["sikar", "vellore", "bidar", "bongaigaon", "eluru[25]", "danapur"],
        "state": ["rajasthan", "tamil_nadu", "karnataka", "assam", "andhra_pradesh", "bihar"],
        "current_job_yrs": [3, 0, 5, 10, 6, 1],
        "current_house_yrs": [11, 11, 13, 12, 12, 12],
        "job_stability": ["variable", "moderate", "variable", "variable", "moderate", "moderate"],
        "city_tier": ["unknown", "unknown", "unknown", "unknown", "unknown", "tier_3"],
        "state_default_rate": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    })

# Fixture to create pipeline segment for use in tests
@pytest.fixture
def pipeline(): 
    return Pipeline([
        ("feature_scaler_encoder", ColumnTransformer(
            transformers=[
                ("scaler", StandardScaler(), NUMERICAL_COLUMNS), 
                ("nominal_encoder", OneHotEncoder(drop="first", sparse_output=False), ["house_ownership"]),
                ("ordinal_encoder", OrdinalEncoder(categories=ORDINAL_COLUMN_ORDERS), ["job_stability", "city_tier"])  
            ],
            remainder="passthrough",
            verbose_feature_names_out=False
        ).set_output(transform="pandas")),
        ("feature_selector", FeatureSelector(columns_to_keep=COLUMNS_TO_KEEP))
    ])


# --- TestModelPreprocessingPipeline class ---
# Inherits from BasePipelineTests which adds the following integration tests:
# .test_pipeline_can_be_cloned()
# .test_pipeline_fit_transform_equivalence()
# .test_pipeline_fit_and_transform_raise_type_error_if_X_not_df()
# .test_pipeline_transform_raises_not_fitted_error_if_unfitted()
# .test_pipeline_transform_does_not_modify_input_df()
# .test_fitted_pipeline_can_be_pickled()
# .test_pipeline_transform_raises_value_error_for_wrong_column_order()
# .test_pipeline_transform_preserves_df_index()
class TestModelPreprocessingPipeline(BasePipelineTests):
    # Override parent class method
    def test_pipeline_fit_and_transform_raise_type_error_if_X_not_df(self):
        pass  # pipeline starts with ColumnTransformer which accepts non-DataFrame inputs
    