from huggingface_hub import HfApi

# Constants
LOCAL_PIPELINE_PATH = "models/loan_default_rf_pipeline.joblib"  
HF_HUB_REPO = "JensBender/loan-default-prediction-pipeline"  # make sure to create Hugging Face Hub repository first
FILENAME = "loan_default_rf_pipeline.joblib"
TAG = "v1.0"  # version tag

# Initialize Hugging Face API client 
api = HfApi()

# Upload pipeline
print("Uploading pipeline to Hugging Face Hub...")
api.upload_file(
    path_or_fileobj=LOCAL_PIPELINE_PATH,
    repo_id=HF_HUB_REPO,
    path_in_repo=FILENAME,
    repo_type="model"
)
print(f"Successfully uploaded pipeline '{FILENAME}' to Hugging Face Hub repository '{HF_HUB_REPO}'.")

# Add the version tag to the latest commit
print(f"Creating tag '{TAG}'...")
api.create_tag(
    repo_id=HF_HUB_REPO,
    tag=TAG,
    repo_type="model",
    exist_ok=True  # prevents error if the tag already exists
)
print(f"Successfully created tag '{TAG}' in repo '{HF_HUB_REPO}'.")
