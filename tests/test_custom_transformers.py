# Standard library imports
import os
import sys
import warnings

# Third-party library imports
import pytest
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import BaseEstimator, TransformerMixin

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  

# Add the parent directory to the path (for local imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  

# Local imports
from app.custom_transformers import (
   MissingValueChecker 
)


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


# Scikit-learn compatibility
def test_MissingValueChecker_sklearn_compatibility():
   critical_features = ["feature_1", "feature_2"]
   non_critical_features = ["feature_3", "feature_4"]
   transformer = MissingValueChecker(critical_features, non_critical_features)
   return check_estimator(transformer)
