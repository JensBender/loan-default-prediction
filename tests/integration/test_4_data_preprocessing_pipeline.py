# Standard library imports
import os
import sys

# Third-party library imports
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SimpleImputer, StandardScaler, OneHotEncoder, OrdinalEncoder

# Add the parent directory to the path (for local imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local imports
from app.custom_transformers import (
    MissingValueChecker, 
    MissingValueStandardizer, 
    SnakeCaseFormatter, 
    BooleanColumnTransformer, 
    JobStabilityTransformer, 
    CityTierTransformer, 
    StateDefaultRateTargetEncoder,
    FeatureSelector,
    MissingValueError, 
    ColumnMismatchError,
    CategoricalLabelError
)
from app.global_constants import (
    CRITICAL_FEATURES, 
    NON_CRITICAL_FEATURES, 
    COLUMNS_FOR_SNAKE_CASING,
    BOOLEAN_COLUMN_MAPPINGS,
    JOB_STABILITY_MAP,
    CITY_TIER_MAP,
    NUMERICAL_COLUMNS, 
    NOMINAL_COLUMN_CATEGORIES, 
    ORDINAL_COLUMN_ORDERS, 
    COLUMNS_TO_KEEP
)
from tests.integration.base_pipeline_tests import BaseSupervisedPipelineTests
