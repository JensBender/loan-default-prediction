# Standard library imports
import os
import sys

# Third-party library imports
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np

# Add the parent directory to the path (for local imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local imports
from app.custom_transformers import StateDefaultRateTargetEncoder
from tests.base_transformer_tests import BaseSupervisedTransformerTests


# --- Fixtures ---
# Fixture to instantiate JobStabilityTransformer class for use in tests
@pytest.fixture
def transformer():
    return StateDefaultRateTargetEncoder()

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
    })

# Fixture to create y input Series for use in tests
@pytest.fixture
def y_input():
    return pd.Series([0, 1, 0, 0, 1, 0])


# --- TestStateDefaultRateTargetEncoder class ---
# Inherits from BaseSupervisedTransformerTests which adds the following tests:
# .test_fit_returns_self()
# .test_fit_learns_attributes()
# .test_instance_can_be_cloned()
# .test_fit_transform_equivalence()
# .test_transform_does_not_modify_input_df()
# .test_transform_handles_empty_df()
# .test_instance_can_be_pickled()
# .test_fit_raises_type_error_for_invalid_input()
# .test_transform_raises_type_error_for_invalid_input()
# .test_transform_raises_value_error_for_wrong_column_order()
# BaseSupervisedTransformerTests further inherits the following tests from BaseTransformerTests:
# .test_instantiation()
# .test_transform_raises_not_fitted_error_if_unfitted()
class TestStateDefaultRateTargetEncoder(BaseSupervisedTransformerTests):
    # Class instantiation 
    def test_instantiation(self, transformer):
        # First, run the .test_instantiation() method from the grandparent class BaseTransformerTests
        super().test_instantiation(transformer)
        # Then, add assertion specific to the StateDefaultRateTargetEncoder class
        assert isinstance(transformer, StateDefaultRateTargetEncoder)
    
    # Ensure .fit() raises ValueError if input DataFrame is missing the "state" column 
    @pytest.mark.unit
    def test_fit_raises_value_error_for_missing_state_column(self, transformer, X_input, y_input):
        X = X_input.copy()
        X_without_state = X.drop(columns="state")
        y = y_input.copy()
        expected_error_message = "Input X is missing the 'state' column."
        with pytest.raises(ValueError, match=expected_error_message):
            transformer.fit(X_without_state, y)
