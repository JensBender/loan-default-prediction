# Standard library imports
import os
import sys
import warnings

# Third-party library imports
import pytest
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
    critical_features=['feature1', 'feature2']
    non_critical_features=['feature3', 'feature4']
    transformer = MissingValueChecker(critical_features, non_critical_features)
    assert isinstance(transformer, BaseEstimator)
    assert isinstance(transformer, TransformerMixin)
    assert transformer.critical_features == critical_features
    assert transformer.non_critical_features == non_critical_features