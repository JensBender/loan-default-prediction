import pytest
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal
import numpy as np
import pickle


# Base tests for a sklearn pipeline or subsegment of a pipeline (that individual integration test classes can inherit from)
class BasePipelineTests:
    pass