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
from app.custom_transformers import JobStabilityTransformer
from app.global_constants import JOB_STABILITY_MAP
from tests.base_transformer_tests import BaseTransformerTests


# --- Fixtures ---
# Fixture to instantiate JobStabilityTransformer class for use in tests
@pytest.fixture
def transformer():
    return JobStabilityTransformer(job_stability_map=JOB_STABILITY_MAP)

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
    })


# --- TestJobStabilityTransformer class ---
# Inherits from BaseTransformerTests which adds the following tests:
# .test_instantiation()
# .test_fit_returns_self()
# .test_fit_learns_attributes()
# .test_instance_can_be_cloned()
# .test_fit_transform_equivalence()
# .test_transform_does_not_modify_input_df()
# .test_transform_handles_empty_df()
# .test_instance_can_be_pickled()
# .test_fit_raises_type_error_for_invalid_input()
# .test_transform_raises_not_fitted_error_if_unfitted()
# .test_transform_raises_type_error_for_invalid_input()
# .test_transform_raises_value_error_for_wrong_column_order()
class TestJobStabilityTransformer(BaseTransformerTests):
    # Class instantiation 
    def test_instantiation(self, transformer):
        # First, run the .test_instantiation() method from the parent class BaseTransformerTests
        super().test_instantiation(transformer)
        # Then, add assertions specific to the BooleanColumnTransformer class
        assert isinstance(transformer, JobStabilityTransformer)
        assert transformer.job_stability_map == JOB_STABILITY_MAP

    # Ensure __init__() raises TypeError for invalid "job_stability_map" data type (must be a dictionary)
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_job_stability_map", [
        "a string",
        ["a", "list"],
        ("a", "tuple"),
        {"a", "set"}, 
        1,
        1.23,
        False,
        None
    ])
    def test_init_raises_type_error_for_invalid_job_stability_map(self, invalid_job_stability_map):
        expected_error_message = "'job_stability_map' must be a dictionary specifying the mappings from 'profession' to 'job_stability'."
        with pytest.raises(TypeError, match=expected_error_message):
            JobStabilityTransformer(job_stability_map=invalid_job_stability_map)

    # Ensure __init__() raises ValueError for empty "job_stability_map" dictionary
    @pytest.mark.unit
    def test_init_raises_value_error_for_empty_job_stability_map(self):
        expected_error_message = "'job_stability_map' cannot be an empty dictionary. It must specify the mappings from 'profession' to 'job_stability'."
        with pytest.raises(ValueError, match=expected_error_message):
            JobStabilityTransformer(job_stability_map={}) 

    # Ensure .fit() raises ValueError if input DataFrame is missing the "profession" column 
    @pytest.mark.unit
    def test_fit_raises_value_error_for_missing_profession_column(self, transformer, X_input):
        X = X_input.copy()
        X_without_profession = X.drop(columns="profession")
        expected_error_message = "Input X is missing the 'profession' column."
        with pytest.raises(ValueError, match=expected_error_message):
            transformer.fit(X_without_profession)    

    # Ensure .fit() ignores missing values in "profession" column
    @pytest.mark.unit
    @pytest.mark.parametrize("missing_value", [None, np.nan])
    def test_fit_ignores_missing_values_in_profession_column(self, transformer, missing_value):
        X_with_missing_value = pd.DataFrame({
            "profession": ["artist", missing_value, "web_designer"]
        })  
        # .fit() should not raise an error 
        transformer.fit(X_with_missing_value)
        # Ensure the learned feature number and names are same as in input DataFrame
        assert transformer.n_features_in_ == X_with_missing_value.shape[1]
        assert transformer.feature_names_in_ == X_with_missing_value.columns.tolist()

    # Ensure .transform() successfully converts professions to job stability tiers
    @pytest.mark.unit
    def test_transform_converts_professions_to_job_stability_tiers(self, transformer, X_input):
        X = X_input.copy()
        # Fit and transform
        transformer.fit(X)
        X_transformed = transformer.transform(X)
        # Create expected output
        expected_X_transformed = X_input.copy()
        expected_X_transformed["job_stability"] = ["variable", "moderate", "variable", "variable", "moderate", "moderate"]
        # Ensure actual and expected output DataFrames are identical
        assert_frame_equal(X_transformed, expected_X_transformed)

    # Ensure .transform() assigns "moderate" job stability to professions not in job_stability_map 
    @pytest.mark.unit
    def test_transform_assigns_moderate_to_unmapped_professions(self, transformer):
        X_with_unmapped_professions = pd.DataFrame({
            "profession": ["artist", "unmapped_profession_1", "web_designer", "unmapped_profession_2"]
        })      
        # Fit and transform
        transformer.fit(X_with_unmapped_professions)
        X_transformed = transformer.transform(X_with_unmapped_professions)
        # Create expected output
        expected_X_transformed = pd.DataFrame({
            "profession": ["artist", "unmapped_profession_1", "web_designer", "unmapped_profession_2"],
            "job_stability": ["variable", "moderate", "variable", "moderate"]  # .fillna("moderate")
        })  
        # Ensure actual and expected output DataFrames are identical
        assert_frame_equal(X_transformed, expected_X_transformed)

    # Ensure .transform() assigns "moderate" job stability for missing values in profession column
    @pytest.mark.unit
    @pytest.mark.parametrize("missing_value", [None, np.nan])
    def test_transform_assigns_moderate_to_missing_professions(self, transformer, missing_value):
        X_with_missing_value = pd.DataFrame({
            "profession": ["artist", missing_value, "web_designer"]
        })        
        # Fit and transform
        transformer.fit(X_with_missing_value)
        X_transformed = transformer.transform(X_with_missing_value)
        # Create expected output
        expected_X_transformed = pd.DataFrame({
            "profession": ["artist", missing_value, "web_designer"],
            "job_stability": ["variable", "moderate", "variable"]  # .fillna("moderate")
        }) 
        # Ensure actual and expected output DataFrames are identical
        assert_frame_equal(X_transformed, expected_X_transformed) 

    # Ensure .transform() ignores other columns 
    @pytest.mark.unit
    def test_transform_ignores_other_columns(self, transformer, X_input):
        X = X_input.copy()
        # Create DataFrame of other columns
        other_columns = [column for column in X.columns if column != "profession"]
        X_without_profession = X[other_columns].copy()
        # Fit and transform on entire DataFrame
        transformer.fit(X)
        X_transformed = transformer.transform(X)
        # Create transformed DataFrame of other columns
        X_transformed_without_profession = X_transformed[other_columns]
        # Ensure untransformed and transformed DataFrames of other columns are identical
        assert_frame_equal(X_without_profession, X_transformed_without_profession)

    # Ensure .transform() raises TypeError for "profession" values with invalid data types (must be strings or missing values)
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_profession_data_type", [
        ["a", "list"],
        ("a", "tuple"),
        {"a": "dictionary"},
        {"a", "set"}, 
        1,
        1.23,
        False
    ])
    def test_transform_raises_type_error_for_invalid_profession(self, transformer, invalid_profession_data_type):
        X = pd.DataFrame({
            "profession": ["artist", "computer_hardware_engineer", "web_designer"]
        })
        transformer.fit(X)
        X_with_invalid_profession = pd.DataFrame({
            "profession": ["artist", invalid_profession_data_type, "web_designer"]
        })
        expected_error_message = "All values in 'profession' column must be strings or missing values."
        with pytest.raises(TypeError, match=expected_error_message):
            transformer.transform(X_with_invalid_profession)  
    