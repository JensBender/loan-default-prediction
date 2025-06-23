# Standard library imports
import os
import sys
import warnings

# Third-party library imports
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
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


# --- Test MissingValueChecker class ---
# Class instantiation 
def test_MissingValueChecker_instantiation():
   transformer = MissingValueChecker(CRITICAL_FEATURES, NON_CRITICAL_FEATURES)
   assert isinstance(transformer, BaseEstimator)
   assert isinstance(transformer, TransformerMixin)
   assert transformer.critical_features == CRITICAL_FEATURES
   assert transformer.non_critical_features == NON_CRITICAL_FEATURES


# --- Test FeatureSelector class ---
# Ensure .fit() returns the instance (self)
def test_feature_selector_fit_returns_self(sample_df):
   feature_selector = FeatureSelector(columns_to_keep=COLUMNS_TO_KEEP)
   fitted_feature_selector = feature_selector.fit(sample_df)
   assert fitted_feature_selector == feature_selector

# Ensure equal output of .fit().transform() and .fit_transform()
def test_feature_selector_fit_transform_equivalence(sample_df):
   feature_selector_1 = FeatureSelector(COLUMNS_TO_KEEP)
   feature_selector_2 = FeatureSelector(COLUMNS_TO_KEEP)
   X_fit_then_transform = feature_selector_1.fit(sample_df).transform(sample_df) 
   X_fit_transform = feature_selector_2.fit_transform(sample_df)
   assert_frame_equal(X_fit_then_transform, X_fit_transform)
