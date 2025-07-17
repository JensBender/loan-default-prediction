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
from app.custom_transformers import StateDefaultRateTargetEncoder, MissingValueError
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
# .test_transform_raises_value_error_for_extra_column()
# .test_transform_raises_value_error_for_wrong_column_order()
# .test_transform_preserves_df_index()
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
        y = y_input.copy()
        X_without_state = X.drop(columns="state")
        expected_error_message = "Input X is missing the 'state' column."
        with pytest.raises(ValueError, match=expected_error_message):
            transformer.fit(X_without_state, y)

    # Ensure .fit() raises MissingValueError for missing values in the "state" column
    @pytest.mark.unit
    @pytest.mark.parametrize("missing_value", [None, np.nan])
    def test_fit_raises_missing_value_error_for_missing_states(self, transformer, missing_value):
        X_with_missing_state = pd.DataFrame({
            "state": ["state_1", missing_value, "state_2"]
        })
        y = pd.Series([0, 0, 1])  
        expected_error_message = "'state' column cannot contain missing values."
        with pytest.raises(MissingValueError, match=expected_error_message):
            transformer.fit(X_with_missing_state, y)

    # Ensure .fit() raises TypeError for non-string values in the "state" column
    @pytest.mark.unit
    @pytest.mark.parametrize("non_string_value", [
        ["a", "list"],
        ("a", "tuple"),
        {"a", "set"},
        {"a": "dictionary"}, 
        1,
        1.23,
        False
    ])
    def test_fit_raises_type_error_for_non_string_states(self, transformer, non_string_value):
        X_with_non_string_state = pd.DataFrame({
            "state": ["state_1", non_string_value, "state_2"]
        })  
        y = pd.Series([0, 0, 1])  
        expected_error_message = "All values in 'state' column must be strings."
        with pytest.raises(TypeError, match=expected_error_message):
            transformer.fit(X_with_non_string_state, y)    

    # Ensure .fit() raises ValueError for unequal index of X and y
    @pytest.mark.unit
    @pytest.mark.parametrize("X, y", [
        (
            pd.DataFrame({"state": ["state_1", "state_2", "state_3"]}, index=[1, 2, 3]), 
            pd.Series([0, 0, 1], index=[4, 5, 6])
        ),
        (
            pd.DataFrame({"state": ["state_1", "state_2", "state_3"]}, index=[1, 2, 3]), 
            pd.Series([0, 1], index=[1, 2])
        )
    ])
    def test_fit_raises_value_error_for_unequal_index_of_X_and_y(self, transformer, X, y):
        expected_error_message = "Input X and y must have the same index."
        with pytest.raises(ValueError, match=expected_error_message):
            transformer.fit(X, y)

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
    
    # Ensure .fit() raises MissingValueError for missing values on y input 

    # Ensure .transform() raises TypeError for non-string values in the "state" column
    @pytest.mark.unit
    @pytest.mark.parametrize("non_string_value", [
        ["a", "list"],
        ("a", "tuple"),
        {"a", "set"},
        {"a": "dictionary"}, 
        1,
        1.23,
        False
    ])
    def test_transform_raises_type_error_for_non_string_states(self, transformer, non_string_value):
        X = pd.DataFrame({
            "state": ["state_1", "state_2", "state_3"]
        }) 
        X_with_non_string_state = pd.DataFrame({
            "state": ["state_1", non_string_value, "state_3"]
        })  
        y = pd.Series([0, 0, 1])  
        # Fit on original DataFrame, but transform on DataFrame with non-string state 
        transformer.fit(X, y)
        expected_error_message = "All values in 'state' column must be strings."
        with pytest.raises(TypeError, match=expected_error_message):
            transformer.transform(X_with_non_string_state)

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

    # Ensure .transform() ignores other columns 
    @pytest.mark.unit
    def test_transform_ignores_other_columns(self, transformer, X_input, y_input):
        X = X_input.copy()
        y = y_input.copy()
        # Create DataFrame of other columns
        other_columns = [column for column in X.columns if column != "state"]
        X_without_state = X[other_columns].copy()
        # Fit and transform on entire DataFrame
        transformer.fit(X, y)
        X_transformed = transformer.transform(X)
        # Create transformed DataFrame of other columns
        X_transformed_without_city = X_transformed[other_columns]
        # Ensure untransformed and transformed DataFrames of other columns are identical
        assert_frame_equal(X_without_state, X_transformed_without_city)

    # Ensure .transform() raises TypeError for "state" values with invalid data type (must be strings or missing values)
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_state_data_type", [
        ["a", "list"],
        ("a", "tuple"),
        {"a": "dictionary"},
        {"a", "set"}, 
        1,
        1.23,
        False
    ])
    def test_transform_raises_type_error_for_invalid_state(self, transformer, invalid_state_data_type):
        # X and y input
        X = pd.DataFrame({
            "state": ["state_1", "state_1", "state_2", "state_2"]
        })
        y = pd.Series([0, 1, 1, 1], name="default")
        # Fit 
        transformer.fit(X, y)
        # Transfrom on DataFrame with invalid data type of a single "state" value
        X_with_invalid_state = pd.DataFrame({
            "state": ["state_1", "state_2", invalid_state_data_type]
        })
        expected_error_message = "All values in 'state' column must be strings or missing values."
        with pytest.raises(TypeError, match=expected_error_message):
            transformer.transform(X_with_invalid_state) 
