## Loan Default Prediction Pipeline

**Model Details**  
Author: Jens Bender  
License: Apache 2.0  
Model: Random Forest Classifier  
Version: 1.0  
Language: Python  
Framework: scikit-learn  
Task: Binary classification  
Input: Customer loan application data (tabular)  
Output: Predicted probability of loan default

**Intended Use**  
Predict the probability of loan default for applicants to help financial institutions manage credit risk.  
Not intended for use in production without additional model validation and fairness assessment. Predictions should not be used as the sole basis for loan approval. Further bias, fairness, and explainability analysis is recommended before deployment.

**Training Data**  
Dataset: [Loan Prediction Based on Customer Behavior (Kaggle)](https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior)  
- 252,000 samples, 11 features  
- 12.3% default rate  
- Features include demographic, financial, and location-based attributes

**Model Evaluation**  
| Metric | Value |
|---------|--------|
| AUC-PR | 0.62 |
| Recall (class 1) | 0.80 |
| Precision (class 1) | 0.54 |
| F1-score (class 1) | 0.64 |
| Accuracy | 0.89 |

**Model Architecture**  
Hyperparameters of Random Forest Classifier:  
- `n_estimators=225`  
- `max_depth=26`  
- `min_samples_split=2`  
- `min_samples_leaf=1`  
- `max_features=0.13`  
- `class_weight='balanced'`

End-to-end `scikit-learn` pipeline containing preprocessing and a Random Forest classifier model. The optimized decision threshold (0.29) is applied in post-processing during deployment, not within the pipeline itself.

**Deployment**  
The model pipeline is serialized via `joblib` and deployed as a Dockerized web app with FastAPI backend and Gradio frontend, hosted on Hugging Face Spaces.

**Links**
| Repository | Link |
|----------|------|
| Code | [github.com/JensBender/loan-default-prediction](https://github.com/JensBender/loan-default-prediction) |
| Pipeline | [huggingface.co/JensBender/loan-default-prediction-pipeline](https://huggingface.co/JensBender/loan-default-prediction-pipeline) |
| App | [huggingface.co/JensBender/loan-default-prediction-app](https://huggingface.co/spaces/JensBender/loan-default-prediction-app) |
