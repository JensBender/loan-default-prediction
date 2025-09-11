from pathlib import Path


# Helper function to get path to root directory 
def get_root_directory(anchor_file: str = "pytest.ini") -> Path:
    file_path = Path(__file__).resolve()
    for parent in file_path.parents:
        if (parent / anchor_file).exists():
            return parent
    raise FileNotFoundError(f"Root directory not found: anchor file '{anchor_file}' is missing.")
