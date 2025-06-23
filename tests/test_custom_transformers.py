# Standard library imports
import os
import sys
import warnings

# Third-party library imports
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
import pandas as pd
from pandas.testing import assert_frame_equal

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  

# Add the parent directory to the path (for local imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  

# Local imports
from app.custom_transformers import (
   MissingValueChecker,
   FeatureSelector
)

from app.global_constants import (
   CRITICAL_FEATURES,
   NON_CRITICAL_FEATURES,
   BOOLEAN_COLUMN_MAPPINGS,
   JOB_STABILITY_MAP,
   CITY_TIER_MAP,
   COLUMNS_TO_KEEP
)


# --- Fixtures ---
# Fixture to provide sample pandas DataFrame that mirrors real data for testing
@pytest.fixture
def sample_df():
   return pd.DataFrame({
      "income": [9121364, 2636544, 9470213, 6558967, 6245331, None],
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

# Sample X input DataFrame for FeatureSelector()
@pytest.fixture
def X_input_for_feature_selector():
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


# --- Test MissingValueChecker class ---
# Class instantiation 
def test_missing_value_checker_instantiation():
   transformer = MissingValueChecker(CRITICAL_FEATURES, NON_CRITICAL_FEATURES)
   assert isinstance(transformer, BaseEstimator)
   assert isinstance(transformer, TransformerMixin)
   assert transformer.critical_features == CRITICAL_FEATURES
   assert transformer.non_critical_features == NON_CRITICAL_FEATURES


# --- Test FeatureSelector class ---
# Class instantiation 
def test_feature_selector_instantiation():
   feature_selector = FeatureSelector(columns_to_keep=COLUMNS_TO_KEEP)   
   assert isinstance(feature_selector, BaseEstimator)
   assert isinstance(feature_selector, TransformerMixin)
   assert feature_selector.columns_to_keep == COLUMNS_TO_KEEP

# Ensure .fit() returns the instance (self)
def test_feature_selector_fit_returns_self(X_input_for_feature_selector):
   X = X_input_for_feature_selector.copy()
   feature_selector = FeatureSelector(columns_to_keep=COLUMNS_TO_KEEP)
   fitted_feature_selector = feature_selector.fit(X)
   assert fitted_feature_selector == feature_selector

# Ensure equal output of .fit().transform() and .fit_transform()
def test_feature_selector_fit_transform_equivalence(X_input_for_feature_selector):
   X = X_input_for_feature_selector.copy()
   feature_selector_1 = FeatureSelector(COLUMNS_TO_KEEP)
   feature_selector_2 = FeatureSelector(COLUMNS_TO_KEEP)
   X_fit_then_transform = feature_selector_1.fit(X).transform(X) 
   X_fit_transform = feature_selector_2.fit_transform(X)
   assert_frame_equal(X_fit_then_transform, X_fit_transform)

# Ensure transform() is idempotent, i.e., calling it multiple times with the same input returns the same output
def test_feature_selector_transform_is_idempotent(X_input_for_feature_selector):
   X = X_input_for_feature_selector.copy()
   feature_selector = FeatureSelector(COLUMNS_TO_KEEP)
   feature_selector.fit(X)
   X_transformed_1 = feature_selector.transform(X)
   X_transformed_2 = feature_selector.transform(X)
   assert_frame_equal(X_transformed_1, X_transformed_2)

# Ensure transform() outputs DataFrame with correct number of rows and columns
def test_feature_selector_transform_output_shape(X_input_for_feature_selector):
   X = X_input_for_feature_selector.copy()
   expected_n_rows = X.shape[0]
   expected_n_columns = len(COLUMNS_TO_KEEP)
   expected_shape = (expected_n_rows, expected_n_columns)
   feature_selector = FeatureSelector(COLUMNS_TO_KEEP)
   X_transformed = feature_selector.fit_transform(X)
   assert X_transformed.shape == expected_shape

# Ensure transformer can be cloned, which is important for sklearn Pipeline compatibility
def test_feature_selector_is_clonable(X_input_for_feature_selector):
   X = X_input_for_feature_selector.copy()
   feature_selector = FeatureSelector(COLUMNS_TO_KEEP)
   feature_selector.fit(X)
   cloned_feature_selector = clone(feature_selector)
   # Check that it's a new object, not a pointer to the old one
   assert cloned_feature_selector is not feature_selector
   # Check that the clone has the same parameters
   assert cloned_feature_selector.get_params() == feature_selector.get_params()
   # Check that modifying the clone's parameters (add column to list)...
   cloned_feature_selector.columns_to_keep.append("added_column")
   # ... doesn't change the original's parameters
   assert feature_selector.columns_to_keep == COLUMNS_TO_KEEP

# Ensure transform() returns DataFrame with expected columns
def test_feature_selector_transform_returns_expected_df(X_input_for_feature_selector):
   X = X_input_for_feature_selector.copy()
   X_expected = X[COLUMNS_TO_KEEP].copy()
   feature_selector = FeatureSelector(COLUMNS_TO_KEEP)
   X_transformed = feature_selector.fit_transform(X)
   # Check that the output is a pandas DataFrame
   assert isinstance(X_transformed, pd.DataFrame)
   # Check that the column names of the output DataFrame are as expected
   assert list(X_transformed.columns) == list(COLUMNS_TO_KEEP)
   # Check that the content and structure of the output DataFrame is as expected
   assert_frame_equal(X_transformed, X_expected)