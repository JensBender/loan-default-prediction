# Standard library imports
import os
import sys

# Third-party library imports
import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Add the parent directory to the path (for local imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local imports
from app.custom_transformers import MissingValueChecker, MissingValueError
from app.global_constants import CRITICAL_FEATURES, NON_CRITICAL_FEATURES
