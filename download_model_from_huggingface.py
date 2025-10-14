from huggingface_hub import hf_hub_download

# Name of Hugging Face Hub repository (make sure the repo exists)
hf_hub_repo = "JensBender/loan-default-prediction-pipeline"
# Filename of model pipeline in the Hugging Face Hub repository (make sure the file exists)
filename = "loan_default_rf_pipeline.joblib"
# Local directory in which the file will be saved
local_dir = "./models"

# Download pipeline
print("Downloading pipeline from Hugging Face Hub...")

hf_hub_download(
    repo_id=hf_hub_repo, 
    filename=filename, 
    local_dir=local_dir
)

print(f"Successfully downloaded pipeline '{filename}' from Hugging Face Hub repository '{hf_hub_repo}'.")
