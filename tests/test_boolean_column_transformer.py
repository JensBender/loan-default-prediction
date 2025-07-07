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
from app.custom_transformers import BooleanColumnTransformer, ColumnMismatchError, CategoricalLabelError
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

    # Ensure .fit() raises TypeError if any individual mapping within the boolean_column_mappings is not a dictionary
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_mapping", [
        "a string",
        ["a", "list"],
        ("a", "tuple"),
        1,
        1.23,
        False,
        None
    ])
    def test_fit_raises_type_error_for_invalid_mapping(self, X_input, invalid_mapping):
        X = X_input.copy()
        transformer = BooleanColumnTransformer(boolean_column_mappings={
            "married": {"married": True, "single": False},
            "car_ownership": invalid_mapping    
        })
        expected_error_message = "The mapping for 'car_ownership' must be a dictionary."
        with pytest.raises(TypeError, match=expected_error_message):
            transformer.fit(X)

    # Ensure .fit() raises CategoricalLabelError for labels not specified in the mappings
    @pytest.mark.unit
    @pytest.mark.parametrize("boolean_column, unspecified_label", [
        ("married", "divorced"), 
        ("married", "yes"), 
        ("married", "no"),
        ("car_ownership", "maybe"),
        ("car_ownership", "lamborghini"),
        ("car_ownership", "soon"),
    ])
    def test_fit_raises_categorical_label_error_for_unspecified_labels(self, transformer, boolean_column, unspecified_label):
        X = pd.DataFrame({
            "married": ["single", "married", "single"],
            "car_ownership": ["no", "yes", "no"],
        })
        X.loc[0, boolean_column] = unspecified_label  # modify first row as a representative example
        expected_error_message = f"Column '{boolean_column}' contains labels that are not specified in the mapping: {unspecified_label}."
        with pytest.raises(CategoricalLabelError, match=expected_error_message):
            transformer.fit(X)

    # Ensure .fit() ignores missing values in boolean columns
    @pytest.mark.unit
    @pytest.mark.parametrize("boolean_column", ["married", "car_ownership"])
    @pytest.mark.parametrize("missing_value", [None, np.nan])
    def test_fit_ignores_missing_values_in_boolean_columns(self, transformer, boolean_column, missing_value):
        X = pd.DataFrame({
            "married": ["single", "married", "single"],
            "car_ownership": ["no", "yes", "no"],
        })
        # Modify first row as a representative example
        X.loc[0, boolean_column] = missing_value
        # .fit() should not raise an error 
        try:
            transformer.fit(X)
        except Exception as e:
            pytest.fail(f".fit() raised an unexpected exception for '{missing_value}' in '{boolean_column}' column: {e}")
        # Ensure the learned feature number and names are same as in input DataFrame
        assert transformer.n_features_in_ == X.shape[1]
        assert transformer.feature_names_in_ == X.columns.tolist()
    
    # Ensure .transform() successfully converts categorical string labels to boolean
    @pytest.mark.unit
    def test_transform_converts_categorical_columns_to_boolean(self, transformer, X_input):
        X = X_input.copy()
        # Fit and transform
        transformer.fit(X)
        X_transformed = transformer.transform(X)
        # Expected output
        expected_X_transformed = pd.DataFrame({
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
        # Ensure transformed columns are boolean data type
        assert X_transformed["married"].dtype == "bool"
        assert X_transformed["car_ownership"].dtype == "bool"
        # Ensure actual and expected output DataFrames are identical
        assert_frame_equal(X_transformed, expected_X_transformed)
