import pytest
from sklearn.base import BaseEstimator, TransformerMixin


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