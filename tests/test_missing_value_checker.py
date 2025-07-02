# Standard library imports
import os
import sys

# Third-party library imports
import pytest
import pandas as pd

# Add the parent directory to the path (for local imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local imports
from app.custom_transformers import MissingValueChecker, ColumnMismatchError
from app.global_constants import CRITICAL_FEATURES, NON_CRITICAL_FEATURES
from tests.base_transformer_tests import BaseTransformerTests


# --- Fixtures ---
# Fixture to instantiate MissingValueChecker class for use in tests
@pytest.fixture
def transformer():
    return MissingValueChecker(
        critical_features=CRITICAL_FEATURES, 
        non_critical_features=NON_CRITICAL_FEATURES
    )

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


# --- TestMissingValueChecker class ---
# Inherits from BaseTransformerTests which adds the following tests:
# .test_instantiation()
# .test_fit_returns_self()
# .test_fit_learns_attributes()
# .test_instance_can_be_cloned()
# .test_fit_transform_equivalence()
# .test_instance_can_be_pickled()
# .test_fit_raises_type_error_for_invalid_input()
# .test_transform_raises_not_fitted_error_if_unfitted()
# .test_transform_raises_type_error_for_invalid_input()
# .test_transform_raises_value_error_for_wrong_column_order()
class TestMissingValueChecker(BaseTransformerTests):
    # Class instantiation 
    @pytest.mark.unit
    def test_instantiation(self, transformer):
        # First, run the .test_instantiation() method from the parent class BaseTransformerTests
        super().test_instantiation(transformer)
        # Then, add assertions specific to the MissingValueChecker class
        assert isinstance(transformer, MissingValueChecker)
        assert transformer.critical_features == CRITICAL_FEATURES
        assert transformer.non_critical_features == NON_CRITICAL_FEATURES

    # Ensure __init__() raises TypeError for invalid critical_features data type (must be a list)
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_critical_features", [
        "a string",
        {"a": "dictionary"},
        ("a", "tuple"),
        1,
        1.23,
        False,
        None        
    ])
    def test_init_raises_type_error_for_invalid_critical_features(self, invalid_critical_features):
        expected_error_message = "'critical_features' must be a list of column names."
        with pytest.raises(TypeError, match=expected_error_message):
            MissingValueChecker(invalid_critical_features, NON_CRITICAL_FEATURES)

    # Ensure __init__() raises TypeError for invalid non_critical_features data type (must be a list)
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_non_critical_features", [
        "a string",
        {"a": "dictionary"},
        ("a", "tuple"),
        1,
        1.23,
        False,
        None        
    ])
    def test_init_raises_type_error_for_invalid_non_critical_features(self, invalid_non_critical_features):
        expected_error_message = "'non_critical_features' must be a list of column names."
        with pytest.raises(TypeError, match=expected_error_message):
            MissingValueChecker(CRITICAL_FEATURES, invalid_non_critical_features)

    # Ensure .fit() raises ColumnMismatchError for missing columns in the input DataFrame
    @pytest.mark.unit
    @pytest.mark.parametrize("missing_columns", [
        "income", 
        "age", 
        ["experience", "married"],
        ["house_ownership", "car_ownership"],
        ["profession", "city", "state"],
        ["current_job_yrs", "current_house_yrs"],
    ])
    def test_fit_raises_column_mismatch_error_for_missing_columns(self, transformer, X_input, missing_columns):
        X = X_input.copy()
        X_with_missing_columns = X.drop(columns=missing_columns)
        with pytest.raises(ColumnMismatchError):
            transformer.fit(X_with_missing_columns)

    # Ensure .fit() raises ColumnMismatchError for unexpected columns in the input DataFrame
    @pytest.mark.unit
    @pytest.mark.parametrize("unexpected_columns", [
        ["unexpected_column_1"],
        ["unexpected_column_1", "unexpected_column_2"],
        ["unexpected_column_1", "unexpected_column_2", "unexpected_column_3"]
    ])
    def test_fit_raises_column_mismatch_error_for_unexpected_columns(self, transformer, X_input, unexpected_columns):
        X_with_unexpected_columns = X_input.copy()
        for unexpected_column in unexpected_columns:
            X_with_unexpected_columns[unexpected_column] = 5 
        with pytest.raises(ColumnMismatchError):
            transformer.fit(X_with_unexpected_columns)

