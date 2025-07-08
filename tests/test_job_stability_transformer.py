# Standard library imports
import os
import sys

# Third-party library imports
import pytest
import pandas as pd

# Add the parent directory to the path (for local imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local imports
from app.custom_transformers import JobStabilityTransformer
from app.global_constants import JOB_STABILITY_MAP
from tests.base_transformer_tests import BaseTransformerTests
