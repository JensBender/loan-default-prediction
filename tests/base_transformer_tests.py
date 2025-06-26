import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from pandas.testing import assert_frame_equal
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