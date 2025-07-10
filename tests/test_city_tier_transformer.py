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
from app.custom_transformers import CityTierTransformer
from app.global_constants import CITY_TIER_MAP
from tests.base_transformer_tests import BaseTransformerTests


# --- Fixtures ---
# Fixture to instantiate JobStabilityTransformer class for use in tests
@pytest.fixture
def transformer():
    return CityTierTransformer(city_tier_map=CITY_TIER_MAP)

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
    })


# --- TestCityTierTransformer class ---
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
class TestCityTierTransformer(BaseTransformerTests):
    # Class instantiation 
    def test_instantiation(self, transformer):
        # First, run the .test_instantiation() method from the parent class BaseTransformerTests
        super().test_instantiation(transformer)
        # Then, add assertions specific to the BooleanColumnTransformer class
        assert isinstance(transformer, CityTierTransformer)
        assert transformer.city_tier_map == CITY_TIER_MAP
    
    # Ensure __init__() raises TypeError for invalid "city_tier_map" data type (must be a dictionary)
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_city_tier_map", [
        "a string",
        ["a", "list"],
        ("a", "tuple"),
        {"a", "set"}, 
        1,
        1.23,
        False,
        None
    ])
    def test_init_raises_type_error_for_invalid_city_tier_map(self, invalid_city_tier_map):
        expected_error_message = "'city_tier_map' must be a dictionary specifying the mappings from 'city' to 'city_tier'."
        with pytest.raises(TypeError, match=expected_error_message):
            CityTierTransformer(city_tier_map=invalid_city_tier_map)

    # Ensure __init__() raises ValueError for empty "job_stability_map" dictionary
    @pytest.mark.unit
    def test_init_raises_value_error_for_empty_city_tier_map(self):
        expected_error_message = "'city_tier_map' cannot be an empty dictionary. It must specify the mappings from 'city' to 'city_tier'."
        with pytest.raises(ValueError, match=expected_error_message):
            CityTierTransformer(city_tier_map={})

    # Ensure .fit() raises ValueError if input DataFrame is missing the "city" column 
    @pytest.mark.unit
    def test_fit_raises_value_error_for_missing_city_column(self, transformer, X_input):
        X = X_input.copy()
        X_without_city = X.drop(columns="city")
        expected_error_message = "Input X is missing the 'city' column."
        with pytest.raises(ValueError, match=expected_error_message):
            transformer.fit(X_without_city) 

    # Ensure .fit() ignores missing values in "city" column
    @pytest.mark.unit
    @pytest.mark.parametrize("missing_value", [None, np.nan])
    def test_fit_ignores_missing_values_in_city_column(self, transformer, missing_value):
        X_with_missing_value = pd.DataFrame({
            "city": ["sikar", missing_value, "bidar"]
        })  
        # .fit() should not raise an error 
        transformer.fit(X_with_missing_value)
        # Ensure the learned feature number and names are same as in input DataFrame
        assert transformer.n_features_in_ == X_with_missing_value.shape[1]
        assert transformer.feature_names_in_ == X_with_missing_value.columns.tolist()

    # Ensure .transform() successfully converts cities to city tiers
    @pytest.mark.unit
    def test_transform_converts_cities_to_city_tiers(self, transformer):
        X = pd.DataFrame({
            "city": ["new_delhi", "bhopal", "vijayanagaram", "kolkata", "vijayawada", "bulandshahr"]
        })
        # Fit and transform
        transformer.fit(X)
        X_transformed = transformer.transform(X)
        # Create expected output
        expected_X_transformed = pd.DataFrame({
            "city": ["new_delhi", "bhopal", "vijayanagaram", "kolkata", "vijayawada", "bulandshahr"],
            "city_tier": ["tier_1", "tier_2", "tier_3", "tier_1", "tier_2", "tier_3"]
        })
        # Ensure actual and expected output DataFrames are identical
        assert_frame_equal(X_transformed, expected_X_transformed)

    # Ensure .transform() assigns "unknown" to cities not in "city_tier_map"
    @pytest.mark.unit
    def test_transform_assigns_unknown_to_unmapped_cities(self, transformer):
        X_with_unmapped_cities = pd.DataFrame({
            "city": ["new_delhi", "unmapped_city_1", "vijayanagaram", "unmapped_city_2", "vijayawada", "unmapped_city_3"]
        })      
        # Fit and transform
        transformer.fit(X_with_unmapped_cities)
        X_transformed = transformer.transform(X_with_unmapped_cities)
        # Create expected output
        expected_X_transformed = pd.DataFrame({
            "city": ["new_delhi", "unmapped_city_1", "vijayanagaram", "unmapped_city_2", "vijayawada", "unmapped_city_3"],
            "city_tier": ["tier_1", "unknown", "tier_3", "unknown", "tier_2", "unknown"]
        })  
        # Ensure actual and expected output DataFrames are identical
        assert_frame_equal(X_transformed, expected_X_transformed)

    # Ensure .transform() assigns "unknown" for missing values in "city" column
    @pytest.mark.unit
    @pytest.mark.parametrize("missing_value", [None, np.nan])
    def test_transform_assigns_unknown_to_missing_cities(self, transformer, missing_value):
        X_with_missing_city = pd.DataFrame({
            "city": ["new_delhi", missing_value, "vijayanagaram"]
        })          
        # Fit and transform
        transformer.fit(X_with_missing_city)
        X_transformed = transformer.transform(X_with_missing_city)
        # Create expected output
        expected_X_transformed = pd.DataFrame({
            "city": ["new_delhi", missing_value, "vijayanagaram"],
            "city_tier": ["tier_1", "unknown", "tier_3"] 
        }) 
        # Ensure actual and expected output DataFrames are identical
        assert_frame_equal(X_transformed, expected_X_transformed) 
