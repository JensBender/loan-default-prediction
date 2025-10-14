from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="models/loan_default_rf_pipeline.joblib",
    path_in_repo="loan_default_rf_pipeline.joblib",
    repo_id="JensBender/loan-default-prediction-pipeline",
    repo_type="model"
)
