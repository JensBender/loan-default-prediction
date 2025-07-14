# Standard library imports
import os
import sys

# Third-party library imports
import pytest
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
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

    # Ensure .fit() correctly learns state default rates
    @pytest.mark.unit
    def test_fit_learns_state_default_rates(self, transformer):
        # X and y input
        X = pd.DataFrame({
            "state": ["state_1", "state_1", "state_2", "state_2"]
        })
        y = pd.Series([0, 1, 1, 1], name="default")
        # Fit
        transformer.fit(X, y)
        # Expected "default_rate_by_state_" learned attribute (pd.Series)
        expected_default_rate_by_state_index = pd.Index(["state_1", "state_2"], name="state")
        expected_default_rate_by_state_ = pd.Series([0.5, 1.0], index=expected_default_rate_by_state_index, name="default")
        # Ensure actual and expected "default_rate_by_state_" are identical
        assert_series_equal(transformer.default_rate_by_state_, expected_default_rate_by_state_)
    
    # Ensure .fit() handles missing values on y input 
    @pytest.mark.unit
    @pytest.mark.parametrize("y_with_missing_values, expected_output", [
        ([0, 1, 1, np.nan], [0.5, 1.0]),
        ([0, 1, 1, None], [0.5, 1.0]),
        ([0, np.nan, 1, np.nan], [0.0, 1.0]),
        ([np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan]),
    ])
    def test_fit_handles_missing_y_values(self, transformer, y_with_missing_values, expected_output):
        # X and y input
        X = pd.DataFrame({
            "state": ["state_1", "state_1", "state_2", "state_2"]
        })
        y = pd.Series(y_with_missing_values, name="default")
        # Fit
        transformer.fit(X, y)
        # Expected "default_rate_by_state_" learned attribute (pd.Series)
        expected_default_rate_by_state_index = pd.Index(["state_1", "state_2"], name="state")
        expected_default_rate_by_state_ = pd.Series(expected_output, index=expected_default_rate_by_state_index, name="default")
        # Ensure actual and expected "default_rate_by_state_" are identical
        assert_series_equal(transformer.default_rate_by_state_, expected_default_rate_by_state_)

    # Ensure .transform() successfully adds the "state_default_rate" column 
    @pytest.mark.unit
    def test_transform_adds_state_default_rate_column(self, transformer):
        # X and y input
        X = pd.DataFrame({
            "state": ["state_1", "state_1", "state_2", "state_2"]
        })
        y = pd.Series([0, 1, 1, 1], name="default")
        # Fit and transform
        transformer.fit(X, y)
        X_transformed = transformer.transform(X)
        # Expected output
        expected_X_transformed = pd.DataFrame({
            "state": ["state_1", "state_1", "state_2", "state_2"],
            "state_default_rate": [0.5, 0.5, 1.0, 1.0]
        })
        # Ensure actual and expected output DataFrames are identical
        assert_frame_equal(X_transformed, expected_X_transformed)

    # Ensure .transform() assigns np.nan to unknown states not seen during .fit() 
    @pytest.mark.unit
    def test_transform_assigns_nan_to_unknown_states(self, transformer):
        # X and y input
        X = pd.DataFrame({
            "state": ["state_1", "state_1", "state_2", "state_2"]
        })
        y = pd.Series([0, 1, 1, 1], name="default")
        # Fit 
        transformer.fit(X, y)
        # Transfrom on DataFrame with unknown state
        X_with_unknown_state = pd.DataFrame({
            "state": ["state_1", "state_2", "unknown_state"]
        })
        X_transformed = transformer.transform(X_with_unknown_state)   
        # Expected output with unknown state
        expected_X_transformed = pd.DataFrame({
            "state": ["state_1", "state_2", "unknown_state"],
            "state_default_rate": [0.5, 1, np.nan]
        })  
        # Ensure actual and expected output DataFrames are identical
        assert_frame_equal(X_transformed, expected_X_transformed)
