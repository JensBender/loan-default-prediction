# Loan Default Prediction Pipeline

## Model Details
**Author:** Jens Bender  
**Version:** 1.0  
**Language:** Python  
**Framework:** scikit-learn  
**License:** Apache 2.0  
**Task:** Binary classification (default vs. not default on loan)  
**Input:** Customer loan application data (tabular data)  
**Output:** Predicted probability of loan default

## Intended Use
Predict the probability of loan default for applicants to help financial institutions manage credit risk.  
Not intended for use in production without additional model validation and fairness assessment.

## Training Data
Dataset: [Loan Prediction Based on Customer Behavior (Kaggle)](https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior)  
- 252,000 samples, 11 features  
- 12.3% default rate  
- Features include demographic, financial, and location-based attributes.

## Evaluation Results
| Metric | Value |
|---------|--------|
| AUC-PR | 0.62 |
| Recall (class 1) | 0.80 |
| Precision (class 1) | 0.54 |
| F1-score (class 1) | 0.64 |
| Accuracy | 0.89 |

## Model Architecture
End-to-end `scikit-learn` pipeline containing preprocessing and a Random Forest classifier model with optimized decision threshold of 0.29.

## Hyperparameters
- `n_estimators=225`  
- `max_depth=26`  
- `min_samples_split=2`  
- `min_samples_leaf=1`  
- `max_features=0.13`  
- `class_weight='balanced'`

## Ethical Considerations and Limitations
- Predictions should not be used as the sole basis for loan approval.  
- Further bias, fairness, and explainability analysis is recommended before deployment.

## Deployment
- The model is serialized via `joblib` and deployed as a web app with FastAPI backend and Gradio frontend, hosted on [Hugging Face Spaces](https://huggingface.co/spaces/JensBender/loan-default-prediction-app).
- Pipeline repository: [loan-default-prediction-pipeline](https://huggingface.co/JensBender/loan-default-prediction-pipeline)
