# --- Imports ---
# Third-party library imports
import pytest 

# Local imports
from api.app import load_pipeline


# --- Function .load_pipeline() ---
class TestLoadPipeline:
    def test_raises_file_not_found_error_for_non_existent_file(self):
        with pytest.raises(FileNotFoundError):
            load_pipeline(path="non_existent_file.joblib")
