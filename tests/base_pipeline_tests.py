import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal
import numpy as np
import pickle


# Base tests for a sklearn pipeline or subsegment of a pipeline (that individual integration test classes can inherit from)
class BasePipelineTests:
    # Ensure pipeline .transform() raises ValueError if columns are in different order than during .fit()
    @pytest.mark.integration
    def test_pipeline_transform_raises_value_error_for_wrong_column_order(X_input, pipeline):
        X = X_input.copy()
        # Fit on original DataFrame X
        pipeline.fit(X)
        # Create DataFrame with different column order than during .fit()
        reversed_columns = X.columns[::-1]  # reverse order as an example
        X_with_wrong_column_order = X[reversed_columns]  
        # Ensure .transform() on wrong column order raises ValueError
        expected_error_message = "Feature names and feature order of input X must be the same as during .fit()."
        with pytest.raises(ValueError, match=expected_error_message):
            pipeline.transform(X_with_wrong_column_order)
