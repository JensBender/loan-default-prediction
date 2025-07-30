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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

# Add the root directory to the path for local imports (by going up two levels from current directory in which this file lives)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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
        transformers=[("categorical_imputer", SimpleImputer(strategy="most_frequent").set_output(transform="pandas"), NON_CRITICAL_FEATURES)],
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
            ("scaler", StandardScaler(), NUMERICAL_COLUMNS), 
            ("nominal_encoder", OneHotEncoder(categories=NOMINAL_COLUMN_CATEGORIES, drop="first", sparse_output=False), ["house_ownership"]),
            ("ordinal_encoder", OrdinalEncoder(categories=ORDINAL_COLUMN_ORDERS), ["job_stability", "city_tier"])  
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    ).set_output(transform="pandas")),
    ("feature_selector", FeatureSelector(columns_to_keep=COLUMNS_TO_KEEP))
])


# --- TestDataPreprocessingPipeline class ---
# Inherits from BaseSupervisedPipelineTests which adds the following integration tests:
# .test_pipeline_can_be_cloned()
# .test_pipeline_fit_transform_equivalence()
# .test_pipeline_fit_and_transform_raise_type_error_if_X_not_df()
# .test_pipeline_transform_does_not_modify_input_df()
# .test_fitted_pipeline_can_be_pickled()
# .test_pipeline_transform_raises_value_error_for_wrong_column_order()
# .test_pipeline_transform_preserves_df_index()
# BaseSupervisedPipelineTests further inherits the following test from BasePipelineTests:
# .test_pipeline_transform_raises_not_fitted_error_if_unfitted()
class TestDataPreprocessingPipeline(BaseSupervisedPipelineTests):
    # Ensure data preprocessing pipeline works as expected
    @pytest.mark.integration
    def test_data_preprocessing_pipeline_happy_path(self, X_input, y_input, pipeline):
        X = X_input.copy()
        y = y_input.copy()
        pipeline.fit(X, y)
        X_transformed = pipeline.transform(X)
        expected_X_transformed = pd.DataFrame({
            "income": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            "age": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            "experience": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            "current_job_yrs": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            "current_house_yrs": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            "state_default_rate": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],            
            "house_ownership_owned": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "house_ownership_rented": [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            "job_stability": [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "city_tier": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "married": [False, False, False, True, False, False],
            "car_ownership": [False, False, True, False, False, False],
        })
        # Ensure actual and expected output DataFrames are identical
        assert_frame_equal(X_transformed, expected_X_transformed)

    # Ensure pipeline .fit() and .transform() raise MissingValueError for missing values in critical features
    @pytest.mark.integration
    @pytest.mark.parametrize("method", ["fit", "transform"])
    @pytest.mark.parametrize("missing_value", [None, np.nan, pd.NA])
    @pytest.mark.parametrize("critical_feature", CRITICAL_FEATURES)
    def test_data_preprocessing_pipeline_raises_missing_value_error_for_critical_features(self, X_input, y_input, pipeline, method, missing_value, critical_feature):
        X = X_input.copy()
        y = y_input.copy()
        X_with_missing_value = X_input.copy()
        X_with_missing_value.loc[0, critical_feature] = missing_value  # in first row as representative example
        # Ensure .fit() raises MissingValueError 
        if method == "fit":
            with pytest.raises(MissingValueError):
                pipeline.fit(X_with_missing_value, y)
        # Ensure .transform() raises MissingValueError with expected error message text
        else:
            # Fit on original DataFrame, but transform on DataFrame with missing values
            pipeline.fit(X, y)
            with pytest.raises(MissingValueError):
                pipeline.transform(X_with_missing_value)

    # Ensure pipline .fit() prints warning message and learns mode for missing values in non-critical features
    @pytest.mark.integration
    @pytest.mark.parametrize("method_with_missing", ["fit_and_transform", "transform_only"])
    @pytest.mark.parametrize("missing_value", [None, np.nan, pd.NA])
    @pytest.mark.parametrize("non_critical_feature", NON_CRITICAL_FEATURES)
    def test_data_preprocessing_pipeline_imputes_missing_values_in_non_critical_features(self, X_input, y_input, pipeline, method_with_missing, missing_value, non_critical_feature, capsys):
        X = X_input.copy()
        y = y_input.copy()
        X_with_missing_value = X_input.copy()
        X_with_missing_value.loc[0, non_critical_feature] = missing_value  # use first row as a representative example
        # Both .fit() and .transform() with missing value 
        if method_with_missing == "fit_and_transform":
            pipeline.fit(X_with_missing_value, y)
            X_transformed = pipeline.transform(X_with_missing_value)
            expected_mode = X_with_missing_value[non_critical_feature].mode()[0]
        # .fit() without and .transform() with missing value
        else:  # method_with_missing == "transform_only"
            pipeline.fit(X, y)
            X_transformed = pipeline.transform(X_with_missing_value)
            expected_mode = X[non_critical_feature].mode()[0]
        # Ensure pipeline prints warning message and imputes mode
        assert "Warning" in capsys.readouterr().out
        if non_critical_feature in ["married", "car_ownership"]:
            assert X_transformed.loc[0, non_critical_feature] == BOOLEAN_COLUMN_MAPPINGS[non_critical_feature][expected_mode]
        else:  # non_critical_feature = "house_ownership"
            assert X_transformed.loc[0, "house_ownership_rented"] == 1.0
            assert X_transformed.loc[0, "house_ownership_owned"] == 0.0       

    # Ensure pipeline .fit() and .transform() raise ColumnMismatchError for missing columns 
    @pytest.mark.integration
    @pytest.mark.parametrize("method", ["fit", "transform"])
    @pytest.mark.parametrize("missing_columns", [
        "income", 
        "age", 
        ["experience", "married"],
        ["house_ownership", "car_ownership"],
        ["profession", "city", "state"],
        ["current_job_yrs", "current_house_yrs"],
    ])
    def test_data_preprocessing_pipeline_raises_column_mismatch_error_for_missing_columns(self, X_input, y_input, pipeline, method, missing_columns):
        X = X_input.copy()
        y = y_input.copy()
        X_with_missing_columns = X.drop(columns=missing_columns)
        # Ensure .fit() raises ColumnMismatchError 
        if method == "fit":
            with pytest.raises(ColumnMismatchError):
                pipeline.fit(X_with_missing_columns, y)
        # Ensure .transform() raises ColumnMismatchError 
        else:
            pipeline.fit(X, y)
            with pytest.raises(ColumnMismatchError):
                pipeline.transform(X_with_missing_columns) 

   # Ensure pipeline .fit() and .transform() raise ColumnMismatchError for unexpected columns (not in CRITICAL_FEATURES or NON_CRITICAL_FEATURES)
    @pytest.mark.integration
    @pytest.mark.parametrize("method", ["fit", "transform"])
    @pytest.mark.parametrize("unexpected_columns", [
        ["unexpected_column_1"],
        ["unexpected_column_1", "unexpected_column_2"],
        ["unexpected_column_1", "unexpected_column_2", "unexpected_column_3"]
    ])
    def test_data_preprocessing_pipeline_raises_column_mismatch_error_for_unexpected_columns(self, X_input, y_input, pipeline, method, unexpected_columns):
        X = X_input.copy()
        y = y_input.copy()
        X_with_unexpected_columns = X_input.copy()
        for unexpected_column in unexpected_columns:
            X_with_unexpected_columns[unexpected_column] = 5 
        # Ensure .fit() raises ColumnMismatchError 
        if method == "fit":
            with pytest.raises(ColumnMismatchError):
                pipeline.fit(X_with_unexpected_columns, y)
        # Ensure .transform() raises ColumnMismatchError 
        else:
            pipeline.fit(X, y)
            with pytest.raises(ColumnMismatchError):
                pipeline.transform(X_with_unexpected_columns)
