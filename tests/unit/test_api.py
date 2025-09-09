# --- Imports ---
# Standard library imports
from unittest.mock import patch

# Third-party library imports
import pytest 

# Local imports
from api.app import load_pipeline


# --- Function .load_pipeline() ---
class TestLoadPipeline:
    @patch("api.app.os.path.exists")
    def test_raises_file_not_found_error_for_non_existent_file(self, mock_os_path_exists):
        # Simulate that the file does not exist
        mock_os_path_exists.return_value = False
        # Ensure FileNotFoundError is raised
        with pytest.raises(FileNotFoundError) as exc_info:
            load_pipeline(path="non_existent_file.joblib")
        # Ensure error message is as expected
        error_msg = str(exc_info.value)
        assert "Pipeline file not found at" in error_msg
        assert "non_existent_file.joblib" in error_msg
        # Ensure os.path.exists was called 
        mock_os_path_exists.assert_called_once_with("non_existent_file.joblib")
