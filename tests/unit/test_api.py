# --- Imports ---
# Standard library imports
from unittest.mock import patch, MagicMock

# Third-party library imports
import pytest 

# Local imports
from api.app import load_pipeline


# --- Function .load_pipeline() ---
class TestLoadPipeline:
    @patch("api.app.joblib.load")
    @patch("api.app.os.path.exists")
    def test_happy_path(self, mock_os_path_exists, mock_joblib_load):
        # Simulate that the file exists
        mock_os_path_exists.return_value = True
        # Simulate loaded pipeline instance
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba = MagicMock()
        mock_joblib_load.return_value = mock_pipeline

        # Call .load_pipeline()
        pipeline = load_pipeline(path="some_path.joblib")

        # Ensure loaded pipeline is a mock pipeline with a "predict_proba" attribute
        assert pipeline is mock_pipeline
        assert hasattr(pipeline, "predict_proba")
        # Ensure os.path.exists was called once
        mock_os_path_exists.assert_called_once_with("some_path.joblib")
        # Ensure joblib.load was called once
        mock_joblib_load.assert_called_once_with("some_path.joblib")

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

    @patch("api.app.joblib.load")
    @patch("api.app.os.path.exists")
    def test_raises_runtime_error_if_joblib_load_fails(self, mock_os_path_exists, mock_joblib_load):
        # Simulate that the file exists
        mock_os_path_exists.return_value = True
        # Simulate an error when loading joblib file
        mock_joblib_load.side_effect = Exception("joblib load error")

        # Ensure RuntimeError is raised
        with pytest.raises(RuntimeError):
            load_pipeline("some_path.joblib")

        # Ensure os.path.exists() and joblib.load() were called
        mock_os_path_exists.assert_called_once_with("some_path.joblib")
        mock_joblib_load.assert_called_once_with("some_path.joblib")
    