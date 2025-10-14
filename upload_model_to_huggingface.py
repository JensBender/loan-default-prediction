from huggingface_hub import HfApi

api = HfApi()

# Path to model pipeline on local machine
pipeline_path = "models/loan_default_rf_pipeline.joblib"
# Name of Hugging Face Hub repository (make sure to create repo first)
hf_hub_repo = "JensBender/loan-default-prediction-pipeline"
# Designated name of model pipeline in the Hugging Face Hub repository
pipeline_in_repo = "loan_default_rf_pipeline.joblib"

print(f"Uploading '{pipeline_path}' to the Hugging Face Hub repo '{hf_hub_repo}'...")

api.upload_file(
    path_or_fileobj=pipeline_path,
    repo_id=hf_hub_repo,
    path_in_repo=pipeline_in_repo,
    repo_type="model"
)

print("Upload complete!")
