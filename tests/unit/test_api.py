# Standard library imports
import os
import sys
import warnings

# Third-party library imports
import pytest

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  

# Add project root directory to the path for local imports (by going up two levels from current directory)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  

# Local imports
from api.app import PipelineInput
