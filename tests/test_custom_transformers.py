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


# Fixture to provide sample pandas DataFrame for testing
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "feature_1": [1, 2, 3],
        "feature_2": [4, 5, 6],
        "feature_3": [7, 8, 9]
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
