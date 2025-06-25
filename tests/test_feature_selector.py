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


# Fixture to create instance of FeatureSelector class
@pytest.fixture
def transformer():
    return FeatureSelector(columns_to_keep=COLUMNS_TO_KEEP)


# Define TestFeatureSelector class (which inherits from BaseTransformerTests)
class TestFeatureSelector(BaseTransformerTests):
    pass
