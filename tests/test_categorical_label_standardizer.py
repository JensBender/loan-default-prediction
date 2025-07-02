# Standard library imports
import os
import sys

# Third-party library imports
import pytest
import pandas as pd

# Add the parent directory to the path (for local imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local imports
from app.custom_transformers import CategoricalLabelStandardizer
from tests.base_transformer_tests import BaseTransformerTests


# --- TestCategoricalLabelStandardizer class ---
# Inherits from BaseTransformerTests which adds the following tests:
# .test_instantiation()
# .test_fit_returns_self()
# .test_fit_learns_attributes()
# .test_instance_can_be_cloned()
# .test_fit_transform_equivalence()
# .test_instance_can_be_pickled()
# .test_fit_raises_type_error_for_invalid_input()
# .test_transform_raises_not_fitted_error_if_unfitted()
# .test_transform_raises_type_error_for_invalid_input()
# .test_transform_raises_value_error_for_wrong_column_order()
class TestCategoricalLabelStandardizer(BaseTransformerTests):
    pass