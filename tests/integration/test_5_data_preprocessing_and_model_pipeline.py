# Standard library imports
import os
import sys

# Third-party library imports
import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Add the root directory to the path for local imports (by going up two levels from current directory in which this file lives)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Local imports
from app.custom_transformers import (
    MissingValueChecker, 
    MissingValueStandardizer, 
    RobustSimpleImputer,
    SnakeCaseFormatter, 
    BooleanColumnTransformer, 
    JobStabilityTransformer, 
    CityTierTransformer, 
    StateDefaultRateTargetEncoder,
    RobustStandardScaler,
    RobustOneHotEncoder,
    RobustOrdinalEncoder,
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
    COLUMNS_TO_KEEP,
    RF_BEST_PARAMS
)
from tests.integration.base_pipeline_tests import BaseSupervisedPipelineTests


# --- Fixtures ---
# Fixture to create X input DataFrame for use in tests
@pytest.fixture
def X_input():
    return pd.DataFrame({
        "income": [300000, 300000, 300000, 500000, 500000, 500000],
        "age": [30, 30, 30, 50, 50, 50],
        "experience": [3, 3, 3, 5, 5, 5],
        "married": ["single", "single", "single", "married", "single", "single"],
        "house_ownership": ["rented", "rented", "norent_noown", "rented", "rented", "owned"],
        "car_ownership": ["no", "no", "yes", "no", "no", "no"],
        "profession": ["Artist", "Computer_hardware_engineer", "Web_designer", "Comedian", 
                       "Financial_Analyst", "Statistician"],
        "city": ["Sikar", "Vellore", "Bidar", "Bongaigaon", "Eluru[25]", "Danapur"],
        "state": ["Rajasthan", "Tamil_Nadu", "Karnataka", "Assam", "Andhra_Pradesh", "Bihar"],
        "current_job_yrs": [3, 3, 3, 5, 5, 5],
        "current_house_yrs": [11, 11, 11, 13, 13, 13],
    })

# Fixture to create y input Series for use in tests
@pytest.fixture
def y_input():
    return pd.Series([0, 0, 0, 1, 1, 1])

# Fixture to create the data preprocessing pipeline for use in tests
@pytest.fixture
def pipeline(): 
    return Pipeline([
    ("missing_value_checker", MissingValueChecker(critical_features=CRITICAL_FEATURES, non_critical_features=NON_CRITICAL_FEATURES)),
    ("missing_value_standardizer", MissingValueStandardizer()),
    ("missing_value_handler", ColumnTransformer(
        transformers=[("categorical_imputer", RobustSimpleImputer(strategy="most_frequent").set_output(transform="pandas"), NON_CRITICAL_FEATURES)],
        remainder="passthrough",
        verbose_feature_names_out=False  # preserve input column names instead of adding prefix 
    ).set_output(transform="pandas")),  # output pd.DataFrame instead of np.array 
    ("snake_case_formatter", SnakeCaseFormatter(columns=COLUMNS_FOR_SNAKE_CASING)),
    ("boolean_column_transformer", BooleanColumnTransformer(boolean_column_mappings=BOOLEAN_COLUMN_MAPPINGS)),
    ("job_stability_transformer", JobStabilityTransformer(job_stability_map=JOB_STABILITY_MAP)),
    ("city_tier_transformer", CityTierTransformer(city_tier_map=CITY_TIER_MAP)),
    ("state_default_rate_target_encoder", StateDefaultRateTargetEncoder()),
    ("feature_scaler_encoder", ColumnTransformer(
        transformers=[
            ("scaler", RobustStandardScaler(), NUMERICAL_COLUMNS), 
            ("nominal_encoder", RobustOneHotEncoder(categories=NOMINAL_COLUMN_CATEGORIES, drop="first", sparse_output=False), ["house_ownership"]),
            ("ordinal_encoder", RobustOrdinalEncoder(categories=ORDINAL_COLUMN_ORDERS), ["job_stability", "city_tier"])  
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    ).set_output(transform="pandas")),
    ("feature_selector", FeatureSelector(columns_to_keep=COLUMNS_TO_KEEP)),
    ("rf_classifier", RandomForestClassifier(**RF_BEST_PARAMS, random_state=42)) 
])


# --- TestDataPreprocessingAndModelPipeline class ---
class TestDataPreprocessingAndModelPipeline:
    pass
