# Standard library imports
import os
import sys

# Third-party library imports
import pytest
import pandas as pd

# Add the parent directory to the path (for local imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local imports
from app.custom_transformers import BooleanColumnTransformer, ColumnMismatchError
from app.global_constants import BOOLEAN_COLUMN_MAPPINGS
from tests.base_transformer_tests import BaseTransformerTests


# --- Fixtures ---
# Fixture to instantiate BooleanColumnTransformer class for use in tests
@pytest.fixture
def transformer():
    return BooleanColumnTransformer(boolean_column_mappings=BOOLEAN_COLUMN_MAPPINGS)

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
        "profession": ["artist", "computer_hardware_engineer", "web_designer", "comedian", 
                       "financial_analyst", "statistician"],
        "city": ["sikar", "vellore", "bidar", "bongaigaon", "eluru[25]", "danapur"],
        "state": ["rajasthan", "tamil_nadu", "karnataka", "assam", "andhra_pradesh", "bihar"],
        "current_job_yrs": [3, 0, 5, 10, 6, 1],
        "current_house_yrs": [11, 11, 13, 12, 12, 12],
    })


# --- TestBooleanColumnTransformer class ---
# Inherits from BaseTransformerTests which adds the following tests:
# .test_instantiation()
# .test_fit_returns_self()
# .test_fit_learns_attributes()
# .test_instance_can_be_cloned()
# .test_fit_transform_equivalence()
# .test_transform_does_not_modify_input_df()
# .test_instance_can_be_pickled()
# .test_fit_raises_type_error_for_invalid_input()
# .test_transform_raises_not_fitted_error_if_unfitted()
# .test_transform_raises_type_error_for_invalid_input()
# .test_transform_raises_value_error_for_wrong_column_order()
class TestBooleanColumnTransformer(BaseTransformerTests):
    # Class instantiation 
    def test_instantiation(self, transformer):
        # First, run the .test_instantiation() method from the parent class BaseTransformerTests
        super().test_instantiation(transformer)
        # Then, add assertions specific to the BooleanColumnTransformer class
        assert isinstance(transformer, BooleanColumnTransformer)
        assert transformer.boolean_column_mappings == BOOLEAN_COLUMN_MAPPINGS

    # Ensure __init__() raises TypeError for invalid boolean_column_mappings data type (must be a dictionary)
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_boolean_column_mappings", [
        "a string",
        ["a", "list"],
        ("a", "tuple"),
        1,
        1.23,
        False,
        None
    ])
    def test_init_raises_type_error_for_invalid_boolean_column_mappings(self, invalid_boolean_column_mappings):
        expected_error_message = "'boolean_column_mappings' must be a dictionary specifying the mappings."
        with pytest.raises(TypeError, match=expected_error_message):
            BooleanColumnTransformer(boolean_column_mappings=invalid_boolean_column_mappings)
    
    # Ensure .fit() raises ColumnMismatchError for missing columns in input DataFrame
    @pytest.mark.unit
    @pytest.mark.parametrize("missing_columns", [
        "married", 
        "car_ownership", 
        ["married", "car_ownership"],
    ])
    def test_fit_raises_column_mismatch_error_for_missing_columns(self, transformer, X_input, missing_columns):
        X = X_input.copy()
        X_with_missing_columns = X.drop(columns=missing_columns)
        with pytest.raises(ColumnMismatchError):
            transformer.fit(X_with_missing_columns)
