import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import NotFittedError
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
import pickle


# Base tests for custom sklearn transformer classes (that individual test classes can inherit from)
class BaseTransformerTests:
    # Class instantiation 
    @pytest.mark.unit
    def test_instantiation(self, transformer):
        assert isinstance(transformer, BaseEstimator)
        assert isinstance(transformer, TransformerMixin)

    # Ensure .fit() returns the instance (self)
    @pytest.mark.unit
    def test_fit_returns_self(self, transformer, X_input):
        X = X_input.copy()
        fitted_transformer = transformer.fit(X)
        assert fitted_transformer is transformer

    # Ensure .fit() stores learned attributes correctly (feature number and names of the input DataFrame) 
    @pytest.mark.unit
    def test_fit_learns_attributes(self, transformer, X_input):
        X = X_input.copy()
        transformer.fit(X)  
        assert hasattr(transformer, "n_features_in_")
        assert hasattr(transformer, "feature_names_in_")
        assert transformer.n_features_in_ == X.shape[1]
        assert transformer.feature_names_in_ == X.columns.tolist()

    # Ensure instance can be cloned (important for scikit-learn compatibility)
    @pytest.mark.unit
    def test_instance_can_be_cloned(self, transformer, X_input):
        X = X_input.copy()
        transformer.fit(X)
        cloned_transformer = clone(transformer)
        # Ensure it's a new object, not a pointer to the old one
        assert cloned_transformer is not transformer
        # Ensure the clone has the same parameters
        assert cloned_transformer.get_params() == transformer.get_params()

    # Ensure equal output of .fit().transform() and .fit_transform()
    @pytest.mark.unit
    def test_fit_transform_equivalence(self, transformer, X_input):
        X = X_input.copy()
        # Create two transformer instances
        transformer_1 = clone(transformer)
        transformer_2 = clone(transformer)
        # Ensure they are different objects in memory
        assert transformer_1 is not transformer_2
        # Perform .fit().transform() vs .fit_transform()
        X_fit_then_transform = transformer_1.fit(X).transform(X) 
        X_fit_transform = transformer_2.fit_transform(X)
        # Ensure the output DataFrames are identical
        assert_frame_equal(X_fit_then_transform, X_fit_transform)

    # Ensure .transform() does not modify the X input DataFrame
    @pytest.mark.unit
    def test_transform_does_not_modify_input_df(self, transformer, X_input):
        X_original = X_input.copy()
        transformer.fit(X_original)
        transformer.transform(X_original)
        assert_frame_equal(X_original, X_input)

    # Ensure .transform() correctly handles an empty DataFrame
    @pytest.mark.unit
    def test_transform_handles_empty_df(self, transformer, X_input):
        X = X_input.copy()
        # Create empty DataFrame
        input_columns = X.columns
        input_data_types = X.dtypes.to_dict()
        X_empty = pd.DataFrame(columns=input_columns).astype(input_data_types)
        # Fit and transform on "full" DataFrame
        transformer.fit(X)
        X_transformed = transformer.transform(X)
        # Transform on empty Dataframe     
        X_transformed_empty = transformer.transform(X_empty)
        # Create expected output (based on "full" DataFrame)
        expected_output_columns = X_transformed.columns
        expected_output_data_types = X_transformed.dtypes.to_dict()
        expected_X_transformed_empty = pd.DataFrame(columns=expected_output_columns).astype(expected_output_data_types)
        # Ensure actual and expected output DataFrames are identical
        assert_frame_equal(X_transformed_empty, expected_X_transformed_empty)

    # Ensure instance can be pickled and unpickled without losing its attributes and functionality
    def test_instance_can_be_pickled(self, transformer, X_input):
        X = X_input.copy()
        transformer.fit(X)
        # Pickle and unpickle
        pickled_transformer = pickle.dumps(transformer)
        unpickled_transformer = pickle.loads(pickled_transformer)
        # Ensure hyperparameters are preserved
        assert transformer.get_params() == unpickled_transformer.get_params()
        # Ensure learned attributes are preserved
        assert transformer.n_features_in_ == unpickled_transformer.n_features_in_
        assert transformer.feature_names_in_ == unpickled_transformer.feature_names_in_
        # Ensure that unpickled transformer produces identical output as original
        X_transformed = transformer.transform(X)
        unpickled_X_transformed = unpickled_transformer.transform(X)
        assert_frame_equal(unpickled_X_transformed, X_transformed)

    # Ensure .fit() raises TypeError for invalid input data type (must be a pandas DataFrame)
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_input_X", [
        np.array([[1, 2], [3, 4]]), 
        "a string",
        {"a": "dictionary"},
        ("a", "tuple"),
        1,
        1.23,
        False,
        None
    ])
    def test_fit_raises_type_error_for_invalid_input(self, transformer, invalid_input_X):
        with pytest.raises(TypeError):
            transformer.fit(invalid_input_X)

    # Ensure .transform() raises NotFittedError if instance has not been fitted yet
    def test_transform_raises_not_fitted_error_if_unfitted(self, transformer, X_input):
        X = X_input.copy()
        # .fit() is intentionally not called here
        with pytest.raises(NotFittedError):
            transformer.transform(X)

    # Ensure .transform() raises TypeError for invalid input data type (must be a pandas DataFrame)
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_X_input", [
        np.array([[1, 2], [3, 4]]), 
        "a string",
        {"a": "dictionary"},
        ("a", "tuple"),
        1,
        1.23,
        False,
        None
    ])
    def test_transform_raises_type_error_for_invalid_input(self, transformer, X_input, invalid_X_input):
        transformer.fit(X_input)
        with pytest.raises(TypeError):
            transformer.transform(invalid_X_input)
 
    # Ensure .transform() raises ValueError if columns are in wrong order
    @pytest.mark.unit
    def test_transform_raises_value_error_for_wrong_column_order(self, transformer, X_input):
        X = X_input.copy()
        transformer.fit(X)
        reversed_columns = X.columns[::-1]  # reverse order as an example
        X_with_wrong_column_order = X[reversed_columns]  
        with pytest.raises(ValueError):
            transformer.transform(X_with_wrong_column_order)