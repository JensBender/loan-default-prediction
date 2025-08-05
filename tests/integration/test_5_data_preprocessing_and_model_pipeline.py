# Standard library imports
import os
import sys

# Third-party library imports
import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

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


# --- Test Functions ---
# Ensure pipeline .predict() output is as expected
@pytest.mark.integration
def test_data_preprocessing_and_model_pipeline_predict_output(X_input, y_input, pipeline):
    X = X_input.copy()
    y = y_input.copy()
    pipeline.fit(X, y)
    predict_output = pipeline.predict(X)
    # Ensure output is an array
    assert isinstance(predict_output, np.ndarray)
    # Ensure array has same number of samples as input and is one-dimensional
    assert predict_output.shape == (len(X), )
    # Ensure all values are 0 or 1
    assert np.all(np.isin(predict_output, [0, 1]))

# Ensure pipeline .predict_proba() output is as expected
@pytest.mark.integration
def test_data_preprocessing_and_model_pipeline_predict_proba_output(X_input, y_input, pipeline):
    X = X_input.copy()
    y = y_input.copy()
    pipeline.fit(X, y)
    predict_proba_output = pipeline.predict_proba(X)
    # Ensure output is an array
    assert isinstance(predict_proba_output, np.ndarray)
    # Ensure array has same number of samples as input and two columns (for class 0 and class 1)
    assert predict_proba_output.shape == (len(X), 2)
    # Ensure all values are between 0 and 1
    assert np.all((predict_proba_output >= 0) & (predict_proba_output <= 1))
    # Ensure each row sums up to 1 (or rather close to 1 using np.isclose)
    assert np.all(np.isclose(np.sum(predict_proba_output, axis=1), 1))

# Ensure pipeline .fit(), .predict(), and .predict_proba() raise TypeError if "X" input is not a pandas DataFrame
@pytest.mark.integration
@pytest.mark.parametrize("method", ["fit", "predict", "predict_proba"])
@pytest.mark.parametrize("invalid_X_input", [
    np.array([[1, 2], [3, 4]]), 
    pd.Series([1, 2, 3]),
    "a string",
    ["a", "list"],
    ("a", "tuple"),
    {"a", "set"}, 
    {"a": "dictionary"},
    1,
    1.23,
    False,
    None
])
def test_data_preprocessing_and_model_pipeline_raises_type_error_if_X_not_df(X_input, y_input, pipeline, method, invalid_X_input):
    y = y_input.copy()
    expected_error_message = "Input X must be a pandas DataFrame."
    if method == "fit":
        with pytest.raises(TypeError, match=expected_error_message):
            pipeline.fit(invalid_X_input, y)
    else: 
        X = X_input.copy()
        pipeline.fit(X, y)
        if method == "predict":          
            with pytest.raises(TypeError, match=expected_error_message):
                pipeline.predict(invalid_X_input)
        else:  # method == "predict_proba"
            with pytest.raises(TypeError, match=expected_error_message):
                pipeline.predict_proba(invalid_X_input)

# Ensure pipeline .fit(), .predict(), and .predict_proba() raise ColumnMismatchError for missing columns 
@pytest.mark.integration
@pytest.mark.parametrize("method", ["fit", "predict", "predict_proba"])
@pytest.mark.parametrize("missing_columns", [
    "income", 
    "age", 
    ["experience", "married"],
    ["house_ownership", "car_ownership"],
    ["profession", "city", "state"],
    ["current_job_yrs", "current_house_yrs"],
])
def test_data_preprocessing_and_model_pipeline_raises_column_mismatch_error_for_missing_columns(X_input, y_input, pipeline, method, missing_columns):
    X_with_missing_columns = X_input.copy()
    X_with_missing_columns = X_with_missing_columns.drop(columns=missing_columns)
    y = y_input.copy()
    if method == "fit":
        with pytest.raises(ColumnMismatchError):
            pipeline.fit(X_with_missing_columns, y)
    else:
        X = X_input.copy()
        pipeline.fit(X, y)
        if method == "predict":
            with pytest.raises(ColumnMismatchError):
                pipeline.predict(X_with_missing_columns)
        else:  # method == "predict_proba"
            with pytest.raises(ColumnMismatchError):
                pipeline.predict_proba(X_with_missing_columns)

# Ensure pipeline .fit(), .predict(), and .predict_proba() raise ColumnMismatchError for unexpected columns (not in CRITICAL_FEATURES or NON_CRITICAL_FEATURES)
@pytest.mark.integration
@pytest.mark.parametrize("method", ["fit", "predict", "predict_proba"])
@pytest.mark.parametrize("unexpected_columns", [
    ["unexpected_column_1"],
    ["unexpected_column_1", "unexpected_column_2"],
    ["unexpected_column_1", "unexpected_column_2", "unexpected_column_3"]
])
def test_data_preprocessing_and_model_pipeline_raises_column_mismatch_error_for_unexpected_columns(X_input, y_input, pipeline, method, unexpected_columns):
    X_with_unexpected_columns = X_input.copy()
    for unexpected_column in unexpected_columns:
        X_with_unexpected_columns[unexpected_column] = 5 
    y = y_input.copy()
    if method == "fit":
        with pytest.raises(ColumnMismatchError):
            pipeline.fit(X_with_unexpected_columns, y)
    else:
        X = X_input.copy()
        pipeline.fit(X, y)
        if method == "predict":
            with pytest.raises(ColumnMismatchError):
                pipeline.predict(X_with_unexpected_columns)
        else:  # method == "predict_proba"
            with pytest.raises(ColumnMismatchError):
                pipeline.predict_proba(X_with_unexpected_columns)

# Ensure pipeline .fit(), .predict(), and .predict_proba() raise MissingValueError for missing values in critical features
@pytest.mark.integration
@pytest.mark.parametrize("method", ["fit", "predict", "predict_proba"])
@pytest.mark.parametrize("missing_value", [None, np.nan, pd.NA])
@pytest.mark.parametrize("critical_feature", CRITICAL_FEATURES)
def test_data_preprocessing_and_model_pipeline_raises_missing_value_error_for_critical_features(X_input, y_input, pipeline, method, missing_value, critical_feature):
    X_with_missing_value = X_input.copy()
    X_with_missing_value.loc[0, critical_feature] = missing_value  # in first row as representative example
    y = y_input.copy()
    if method == "fit":
        with pytest.raises(MissingValueError):
            pipeline.fit(X_with_missing_value, y)
    else:
        X = X_input.copy()
        pipeline.fit(X, y)
        if method == "predict":
            with pytest.raises(MissingValueError):
                pipeline.predict(X_with_missing_value)
        else:  # method == "predict_proba"
            with pytest.raises(MissingValueError):
                pipeline.predict_proba(X_with_missing_value)

# Ensure pipeline .predict() and .predict_proba() impute the mode for missing values in non-critical features
@pytest.mark.integration
@pytest.mark.parametrize("method", ["fit", "predict", "predict_proba"])
@pytest.mark.parametrize("missing_value", [None, np.nan, pd.NA])
@pytest.mark.parametrize("non_critical_feature", NON_CRITICAL_FEATURES)
def test_data_preprocessing_and_model_pipeline_imputes_missing_values_in_non_critical_features(X_input, y_input, pipeline, method, missing_value, non_critical_feature, capsys):
    X_with_missing_value = X_input.copy()
    X_with_missing_value.loc[0, non_critical_feature] = missing_value
    y = y_input.copy()
    if method == "fit":
        pipeline.fit(X_with_missing_value, y)
        # Ensure .fit() on missing value prints warning message 
        assert "Warning" in capsys.readouterr().out
        # Ensure .fit() learns mode
        expected_mode = X_with_missing_value[non_critical_feature].mode()[0]
        column_transformer = pipeline.named_steps["missing_value_handler"]
        robust_simple_imputer = column_transformer.named_transformers_["categorical_imputer"]
        feature_index = NON_CRITICAL_FEATURES.index(non_critical_feature)
        learned_mode = robust_simple_imputer.statistics_[feature_index]
        assert learned_mode == expected_mode
    else:
        X = X_input.copy()
        pipeline.fit(X, y)    
        expected_mode = X[non_critical_feature].mode()[0]
        X_with_manual_impute = X_with_missing_value.fillna({non_critical_feature: expected_mode})
        if method == "predict":
            # Ensure .predict() on missing value imputes the mode
            predict_with_manual_impute = pipeline.predict(X_with_manual_impute)
            predict_with_pipeline_impute = pipeline.predict(X_with_missing_value)
            assert_array_equal(predict_with_pipeline_impute, predict_with_manual_impute)
            # Ensure .predict() on missing value prints warning message
            assert "Warning" in capsys.readouterr().out
        else:  # method == "predict_proba"
            # Ensure .predict_proba() on missing value imputes the mode
            predict_proba_with_manual_impute = pipeline.predict_proba(X_with_manual_impute)
            predict_proba_with_pipeline_impute = pipeline.predict_proba(X_with_missing_value)
            assert_array_equal(predict_proba_with_pipeline_impute, predict_proba_with_manual_impute)
            # Ensure .predict_proba() on missing value prints warning message
            assert "Warning" in capsys.readouterr().out

# Ensure pipeline .fit(), .predict(), and .predict_proba() raise CategoricalLabelError for unknown labels
@pytest.mark.integration
@pytest.mark.parametrize("method", ["fit", "predict", "predict_proba"])
@pytest.mark.parametrize("column", ["married", "car_ownership", "profession", "city"])
def test_data_preprocessing_and_model_pipeline_raises_categorical_label_error_for_unknown_labels(X_input, y_input, pipeline, method, column):
    X_with_unknown_label = X_input.copy()
    X_with_unknown_label.loc[0, column] = "unknown_label"  
    y = y_input.copy()
    if method == "fit":
        with pytest.raises(CategoricalLabelError):
            pipeline.fit(X_with_unknown_label, y)
    else:  
        X = X_input.copy()
        pipeline.fit(X, y) 
        if method == "predict":
            with pytest.raises(CategoricalLabelError):
                pipeline.predict(X_with_unknown_label)
        else:  # method == "predict_proba"
            with pytest.raises(CategoricalLabelError):
                pipeline.predict_proba(X_with_unknown_label)

# Ensure pipeline .predict() and .predict_proba() raise CategoricalLabelError for unknown states not seen during .fit() 
@pytest.mark.integration
@pytest.mark.parametrize("method", ["predict", "predict_proba"])
def test_data_preprocessing_and_model_pipeline_raises_categorical_label_error_for_unknown_states(X_input, y_input, pipeline, method):
    X = X_input.copy()
    y = y_input.copy()
    X_with_unknown_state = X_input.copy()
    X_with_unknown_state.loc[0, "state"] = "unknown_state"  
    pipeline.fit(X, y) 
    if method == "predict":
        with pytest.raises(CategoricalLabelError):
            pipeline.predict(X_with_unknown_state)
    else:  # method == "predict_proba"
        with pytest.raises(CategoricalLabelError):
            pipeline.predict_proba(X_with_unknown_state)

# Ensure pipeline .fit(), .predict(), and .predict_proba() raise ValueError for unknown "house_ownership" categories (not in "categories" hyperparameter of OneHotEncoder) 
@pytest.mark.integration
@pytest.mark.parametrize("method", ["fit", "predict", "predict_proba"])
def test_data_preprocessing_and_model_pipeline_raises_value_error_for_unknown_house_ownership_categories(X_input, y_input, pipeline, method):
    X_with_unknown_category = X_input.copy()
    X_with_unknown_category.loc[0, "house_ownership"] = "unknown_category"  
    y = y_input.copy()
    if method == "fit":
        with pytest.raises(ValueError):
            pipeline.fit(X_with_unknown_category, y)
    else:  
        X = X_input.copy()
        pipeline.fit(X, y) 
        if method == "predict":
            with pytest.raises(ValueError):
                pipeline.predict(X_with_unknown_category)
        else:  # method == "predict_proba"
            with pytest.raises(ValueError):
                pipeline.predict_proba(X_with_unknown_category)

# Ensure pipeline .predict() and .predict_proba() raise ColumnMismatchError for wrong feature order, i.e., not the same as during .fit()
@pytest.mark.integration
@pytest.mark.parametrize("method", ["predict", "predict_proba"])
def test_data_preprocessing_and_model_pipeline_raises_column_mismatch_error_for_wrong_feature_order(X_input, y_input, pipeline, method):
    X = X_input.copy()
    y = y_input.copy()
    pipeline.fit(X, y)
    reversed_columns = X.columns[::-1]  # reverse order as an example
    X_with_wrong_column_order = X[reversed_columns]  
    expected_error_message = "Feature names and feature order of input X must be the same as during .fit()."
    with pytest.raises(ColumnMismatchError, match=expected_error_message):
        if method == "predict":
            pipeline.predict(X_with_wrong_column_order)
        else:  # method == "predict_proba"
            pipeline.predict_proba(X_with_wrong_column_order)      

# Ensure pipeline can be cloned 
@pytest.mark.integration
@pytest.mark.parametrize("method", ["predict", "predict_proba"])
def test_data_preprocessing_and_model_pipeline_can_be_cloned(X_input, y_input, pipeline, method):
    X = X_input.copy()
    y = y_input.copy()
    cloned_pipeline = clone(pipeline)
    pipeline.fit(X, y)
    cloned_pipeline.fit(X, y)
    # Ensure it's a new object, not a pointer to the old one
    assert cloned_pipeline is not pipeline
    # Ensure the original and cloned outputs are identical
    if method == "predict":
        pipeline_predict_output = pipeline.predict(X)
        cloned_pipeline_predict_output = cloned_pipeline.predict(X)
        assert_array_equal(pipeline_predict_output, cloned_pipeline_predict_output)
    else:  # method == "predict_proba"
        pipeline_predict_proba_output = pipeline.predict_proba(X)
        cloned_pipeline_predict_proba_output = cloned_pipeline.predict_proba(X)
        assert_array_equal(pipeline_predict_proba_output, cloned_pipeline_predict_proba_output)

# Ensure pipeline output of .fit().predict() and .fit_predict() is equivalent
@pytest.mark.integration
def test_data_preprocessing_and_model_pipeline_fit_predict_equivalence(X_input, y_input, pipeline):
    X = X_input.copy()
    y = y_input.copy()
    pipeline_1 = clone(pipeline)
    pipeline_2 = clone(pipeline)
    assert pipeline_1 is not pipeline_2  # ensure they are different objects in memory
    pipeline_output_fit_then_predict = pipeline_1.fit(X, y).predict(X) 
    pipeline_output_fit_predict = pipeline_2.fit_predict(X, y)
    assert_array_equal(pipeline_output_fit_then_predict, pipeline_output_fit_predict)
