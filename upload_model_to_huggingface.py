from huggingface_hub import HfApi

api = HfApi()

# Path to model pipeline on local machine
pipeline_path = "models/loan_default_rf_pipeline.joblib"
# Name of Hugging Face Hub repository (make sure to create repo first)
hf_hub_repo = "JensBender/loan-default-prediction-pipeline"
# Filename of model pipeline in the Hugging Face Hub repository
filename = "loan_default_rf_pipeline.joblib"

# Upload pipeline
print("Uploading pipeline to Hugging Face Hub...")

api.upload_file(
    path_or_fileobj=pipeline_path,
    repo_id=hf_hub_repo,
    path_in_repo=filename,
    repo_type="model"
)

print(f"Successfully uploaded pipeline '{filename}' to Hugging Face Hub repository '{hf_hub_repo}'.")
