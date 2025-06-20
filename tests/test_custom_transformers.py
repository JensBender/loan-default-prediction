# Standard library imports
import os
import sys
import warnings

# Third-party library imports
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  

# Add the parent directory to the path (for local imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  

# Local imports
from app.custom_transformers import (
   MissingValueChecker,
   FeatureSelector
)


# Fixture to provide sample pandas DataFrame that mirrors the real data for testing
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
   critical_features = ["feature_1", "feature_2"]
   non_critical_features = ["feature_3", "feature_4"]
   transformer = MissingValueChecker(critical_features, non_critical_features)
   assert isinstance(transformer, BaseEstimator)
   assert isinstance(transformer, TransformerMixin)
   assert transformer.critical_features == critical_features
   assert transformer.non_critical_features == non_critical_features


# --- Test FeatureSelector class ---
# Ensure .fit() returns the instance (self)
def test_feature_selector_fit_returns_self(sample_df):
   feature_selector = FeatureSelector(columns_to_keep=["feature_1", "feature_2"])
   fitted_feature_selector = feature_selector.fit(sample_df)
   assert fitted_feature_selector == feature_selector
