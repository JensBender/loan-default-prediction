# Standard library imports
import os
import sys

# Third-party library imports
import pytest
import pandas as pd

# Add the parent directory to the path (for local imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  

# Local imports
from app.custom_transformers import FeatureSelector
from app.global_constants import COLUMNS_TO_KEEP
from tests.base_transformer_tests import BaseTransformerTests


# --- Fixtures ---
# Fixture to instantiate FeatureSelector class for use in tests
@pytest.fixture
def transformer():
    return FeatureSelector(columns_to_keep=COLUMNS_TO_KEEP)

# Fixture to create X input DataFrame for use in tests
@pytest.fixture
def X_input():
    return pd.DataFrame({
      "income": [-0.82, 1.55, -1.47, 0.51, 1.39, -0.61],
      "age": [-0.64, -0.52, -0.29, -0.52, 0.23, 0.46],
      "experience": [-1.68, -0.84, -1.51, -1.68, 1.31, -0.18],
      "current_job_yrs": [-1.73, -0.36, -1.46, -1.73, -0.36, 0.73],
      "current_house_yrs": [-0.71, 0.71, -0.71, 1.43, 0.71, -0.71],
      "state_default_rate": [-1.08, -1.24, 0.32, -0.31, -0.14, -0.38],
      "house_ownership_owned": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
      "house_ownership_rented": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
      "job_stability": [1.0, 0.0, 2.0, 3.0, 0.0, 3.0],
      "city_tier": [0.0, 2.0, 0.0, 1.0, 0.0, 3.0],
      "married": [False, False, True, False, True, False],
      "car_ownership": [False, True, False, False, True, True],
      "profession": ["computer_hardware_engineer", "web_designer", "lawyer", "firefighter", "artist", "librarian"],
      "city": ["vellore", "bidar", "nizamabad", "farrukhabad", "sikar", "hindupur"],
      "state": ["tamil_nadu", "karnataka", "telangana", "uttar_pradesh", "rajasthan", "andhra_pradesh"]
   })


# --- TestFeatureSelector class ---
# Inherits from BaseTransformerTests which adds the following tests:
# .test_instantiation()
# .test_fit_returns_self()
# .test_fit_learns_attributes()
# .test_instance_can_be_cloned()
# .test_fit_transform_equivalence()
# .test_instance_can_be_pickled()
class TestFeatureSelector(BaseTransformerTests):
    # Class instantiation 
    @pytest.mark.unit
    def test_instantiation(self, transformer):
        # First, run the .test_instantiation() method from the parent class BaseTransformerTests
        super().test_instantiation(transformer)  # asserts transformer is BaseEstimator and TransformerMixin
        # Then, add assertions specific to the FeatureSelector class
        assert isinstance(transformer, FeatureSelector)
        assert transformer.columns_to_keep == COLUMNS_TO_KEEP

    # Ensure .fit() ignores extra columns not in columns_to_keep
    @pytest.mark.unit
    def test_fit_ignores_extra_columns(self, transformer, X_input):
        X = X_input.copy()
        X["extra_column"] = "extra_value"  # extra column that is not in COLUMNS_TO_KEEP
        transformer.fit(X)  # should fit without raising an error
        # Ensure the learned feature number and names are same as in input DataFrame
        assert transformer.n_features_in_ == X.shape[1]
        assert transformer.feature_names_in_ == X.columns.tolist()

    # Ensure __init__() raises ValueError for invalid data types of columns_to_keep (must be a list)
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_columns_to_keep", [
        "a string",
        {"a": "dictionary"},
        ("a", "tuple"),
        1,
        1.23,
        False,
        None
    ])
    def test_init_raises_value_error_for_invalid_columns_to_keep(self, invalid_columns_to_keep):
        with pytest.raises(ValueError):
            FeatureSelector(invalid_columns_to_keep)

    # Ensure .fit() raises ValueError for missing columns in the input DataFrame
    @pytest.mark.unit
    @pytest.mark.parametrize("missing_columns", [
        "income", 
        "age", 
        "experience",
        ["current_job_yrs", "current_house_yrs"],
        ["state_default_rate", "house_ownership_owned", "house_ownership_rented"],
        ["job_stability", "city_tier", "married", "car_ownership"]
    ])
    def test_fit_raises_value_error_for_missing_columns(self, transformer, X_input, missing_columns):
        X = X_input.copy()
        X_with_missing_columns = X.drop(columns=missing_columns)
        with pytest.raises(ValueError):
            transformer.fit(X_with_missing_columns)