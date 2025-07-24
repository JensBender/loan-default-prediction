import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal
import numpy as np
import pickle


# Base tests for a sklearn pipeline or subsegment of a pipeline (that individual integration test classes can inherit from)
class BasePipelineTests:
    # Ensure pipeline instance can be cloned 
    @pytest.mark.integration
    def test_pipeline_can_be_cloned(self, pipeline, X_input):
        X = X_input.copy()
        cloned_pipeline = clone(pipeline)
        pipeline.fit(X)
        pipeline_output = pipeline.transform(X)
        cloned_pipeline.fit(X)
        cloned_pipeline_output = cloned_pipeline.transform(X)
        # Ensure it's a new object, not a pointer to the old one
        assert cloned_pipeline is not pipeline
        # Ensure the original and cloned outputs are identical
        assert_frame_equal(pipeline_output, cloned_pipeline_output)

    # Ensure equal output of .fit().transform() and .fit_transform()
    @pytest.mark.integration
    def test_pipeline_fit_transform_equivalence(self, pipeline, X_input):
        X = X_input.copy()
        # Create two transformer instances
        pipeline_1 = clone(pipeline)
        pipeline_2 = clone(pipeline)
        # Ensure they are different objects in memory
        assert pipeline_1 is not pipeline_2
        # Perform .fit().transform() vs .fit_transform()
        X_fit_then_transform = pipeline_1.fit(X).transform(X) 
        X_fit_transform = pipeline_2.fit_transform(X)
        # Ensure the output DataFrames are identical
        assert_frame_equal(X_fit_then_transform, X_fit_transform)

    # Ensure pipeline .transform() raises ValueError if columns are in different order than during .fit()
    @pytest.mark.integration
    def test_pipeline_transform_raises_value_error_for_wrong_column_order(self, pipeline, X_input):
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
