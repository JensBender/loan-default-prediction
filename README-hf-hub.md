---
language:
  - en
license: apache-2.0
library_name: scikit-learn
model_type: RandomForestClassifier
pipeline_tag: tabular-classification
tags:
  - finance
  - credit-risk
  - loan-default
  - tabular-data
  - random-forest
  - joblib
metrics:
  - auc_pr
  - recall
  - precision
  - f1
thumbnail: "https://raw.githubusercontent.com/JensBender/loan-default-prediction/main/images/header-image.webp"
---

# 🏦 Loan Default Prediction Pipeline
## Model Details
### Model Description
Author: Jens Bender  
License: Apache 2.0  
Model Type: Random Forest Classifier  
Version: 1.0  
Language: Python  
Framework: scikit-learn  
Task: Binary classification  
Input: Tabular data (customer loan application)  
Output: Probability of loan default

### Model Sources
| Repository | Link |
|----------|------|
| Code | [github.com/JensBender/loan-default-prediction](https://github.com/JensBender/loan-default-prediction) |
| Pipeline | [huggingface.co/JensBender/loan-default-prediction-pipeline](https://huggingface.co/JensBender/loan-default-prediction-pipeline) |
| App | [huggingface.co/JensBender/loan-default-prediction-app](https://huggingface.co/spaces/JensBender/loan-default-prediction-app) |

---

## Uses
### Direct Use
- Predict the probability of loan default for applicants to help financial institutions manage credit risk.
- Support loan application officers with model-driven insights.

### Out-of-Scope Use
- Not intended for fully automated lending decisions.
- Not suitable for populations or markets not represented in the training data.
- Not intended for use in production without additional model validation and fairness assessment.

---

## Bias, Risks, and Limitations
- The model reflects patterns in historical data, which may contain sociodemographic or geographic biases.
- Such biases could lead to unfair treatment of certain applicant groups.

### Recommendations
- Use the model as a supportive tool, not as the sole basis for loan approval. 
- Perform bias, fairness, and explainability analysis before production deployment.  

---

## Training Details
### Training Data
Dataset: [Loan Prediction Based on Customer Behavior (Kaggle)](https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior)
- Samples: 252,000  
- Default rate: 12.3%  
- Features: 11  
- Feature types: Demographic, financial, and behavioral variables

### Training Procedure
#### Preprocessing
- Handled duplicates, data types, missing values, and outliers.
- Engineered new features: Job stability, city tier, and state default rate.
- Scaled numerical features and encoded categorical features. 
#### Training Hyperparameters
Random Forest Classifier:  
- `n_estimators=225`  
- `max_depth=26`  
- `min_samples_split=2`  
- `min_samples_leaf=1`  
- `max_features=0.13`  
- `class_weight='balanced'`  

---

## Evaluation
### Testing Data and Metrics
Evaluated model performance on a hold-out test set (10% split).

| Metric | Value |
|---------|--------|
| AUC-PR | 0.62 |
| Recall (class 1) | 0.80 |
| Precision (class 1) | 0.54 |
| F1-score (class 1) | 0.64 |
| Accuracy | 0.89 |

---

## Deployment  
- End-to-end `scikit-learn` pipeline containing preprocessing and a Random Forest classifier model. The optimized decision threshold (0.29) is applied in post-processing during deployment, not within the pipeline itself.
- The model pipeline is serialized via `joblib` and deployed as a Dockerized web app with FastAPI backend and Gradio frontend, hosted on Hugging Face Spaces.

---

## License  
The model pipeline with the trained model weights is licensed under [Apache-2.0](https://huggingface.co/JensBender/loan-default-prediction-pipeline/resolve/main/LICENSE).   
The source code of this project, hosted on [GitHub](https://github.com/JensBender/loan-default-prediction), and the web app hosted on [Hugging Face Spaces](https://huggingface.co/spaces/JensBender/loan-default-prediction-app), are licensed under the [MIT License](LICENSE). 

---

## Citation
BibTeX:
```bibtex
@misc{bender_loan_default_prediction_2025,
  author       = {Jens Bender},
  title        = {Loan Default Prediction Pipeline},
  year         = {2025},
  version      = {1.0},
  publisher    = {Hugging Face Hub},
  howpublished = {\url{https://huggingface.co/JensBender/loan-default-prediction-pipeline}},
  note         = {Machine Learning Model Repository},
  license      = {Apache-2.0}
}
