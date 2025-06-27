# Standard library imports
import os
import sys

# Third-party library imports
import pytest

# Add the parent directory to the path (for local imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local imports
from app.custom_transformers import MissingValueChecker
from app.global_constants import CRITICAL_FEATURES, NON_CRITICAL_FEATURES
from tests.base_transformer_tests import BaseTransformerTests


# --- Fixtures ---
# Fixture to instantiate MissingValueChecker class for use in tests
@pytest.fixture
def transformer():
    return MissingValueChecker(
        critical_features=CRITICAL_FEATURES, 
        non_critical_features=NON_CRITICAL_FEATURES
    )

# Fixture to create X input DataFrame for use in tests

# --- TestMissingValueChecker class ---
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
class TestMissingValueChecker(BaseTransformerTests):
    pass