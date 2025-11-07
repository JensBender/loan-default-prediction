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
---

 <img src="images/header-image.webp" alt="Header Image"> 

# 🏦 Loan Default Prediction Pipeline
This model repository contains a `scikit-learn` pipeline for predicting loan defaults from customer application data. The pipeline includes all data preprocessing steps and a Random Forest Classifier model trained on 252,000 loan applications. It is designed to assist financial institutions in making more informed, data-driven lending decisions and managing credit risk. 

## Model Details
### Model Description
The pipeline takes raw loan application data as input (formatted as a `pandas DataFrame`) and performs all necessary preprocessing steps such as feature engineering, scaling, and encoding. The pipeline then uses a Random Forest Classifier model to predict the probability of loan default (as a `NumPy array`). 

| Pipeline | Version | Framework | Task | Input | Output | Author | License |
|---------------|---------|-----------|------|-------|--------|--------|---------|
| Random Forest model with preprocessing | 1.0 | Python, scikit-learn | Binary classification | Tabular data | Predicted probabilities | Jens Bender | Apache 2.0 |

### Model Sources
| Component | Link |
|----------|------|
| Source Code | [GitHub](https://github.com/JensBender/loan-default-prediction) |
| Model Pipeline | [Hugging Face Hub](https://huggingface.co/JensBender/loan-default-prediction-pipeline) |
| Web App | [Hugging Face Spaces](https://huggingface.co/spaces/JensBender/loan-default-prediction-app) |

### How to Get Started with the Model
#### Using the Web App
The model pipeline is deployed as a web application on [Hugging Face Spaces](https://huggingface.co/spaces/JensBender/loan-default-prediction-app). You can interact with the model directly through the web interface without any installation or coding required.

#### Downloading and Using the Pipeline
The pipeline is serialized as a `joblib` file. You can download and use it for inference with the `huggingface_hub` library as shown below. The optimized decision threshold of 0.29 is not part of the pipeline itself and has to be applied in post-processing.

```python
from huggingface_hub import hf_hub_download
import joblib
import pandas as pd

# Download the pipeline from Hugging Face Hub and load it into memory
pipeline_path = hf_hub_download("JensBender/loan-default-prediction-pipeline", "loan_default_rf_pipeline.joblib")
pipeline = joblib.load(pipeline_path)

# Create a sample DataFrame
# Note: The column names and data types must match the training data
applicant_data = pd.DataFrame({
    "income": [300000],
    "age": [30],
    "experience": [3],
    "married": ["single"],
    "house_ownership": ["rented"],
    "car_ownership": ["no"],
    "profession": ["Artist"],
    "city": ["Sikar"],
    "state": ["Rajasthan"],
    "current_job_yrs": [3],
    "current_house_yrs": [11],
})

# Get predicted probabilities of both classes (0: no default, 1: default)
probabilities = pipeline.predict_proba(applicant_data) 
default_probability = probabilities[0, 1]  # row 0, column 1 of np.ndarray
print(f"Probability of default: {default_probability:.2f}")

# Apply optimized threshold to make a classification decision
threshold = 0.29
prediction = "Default" if default_probability >= threshold else "No Default"
print(f"Threshold: {threshold}")
print(f"Prediction: {prediction}")
```

---

## Uses
### Direct Use
The model is intended to be used as a tool to support credit risk assessment. It can be integrated into decision-making workflows to provide a quantitative measure of default risk for loan applicants.

### Out-of-Scope Use
This model is **not** intended for:
- Fully automated lending decisions without human oversight. The model's predictions should not be the sole factor in any financial decision.
- Evaluating applicants from demographic, geographic, or socioeconomic backgrounds not represented in the training data.
- Use in a production environment without rigorous, ongoing validation and fairness audits. 

---

## Bias, Risks, and Limitations
The model was trained on historical data, which may carry inherent biases related to socioeconomic status, geography, or other demographic factors. Consequently, the model may produce predictions that unfairly disadvantage certain groups of applicants.

### Recommendations
- **Human in the Loop:** Always use this model as part of a broader decision-making framework that includes human oversight.
- **Fairness and Bias Audits:** Before deploying this model in a production environment, conduct thorough fairness and bias analyses to ensure it performs equally across different demographic groups.
- **Model Monitoring:** Continuously monitor the model's performance and predictions to detect and mitigate any performance degradation or emerging biases.

---

## Training Details
### Training Data
The model was trained on the "Loan Prediction Based on Customer Behavior" dataset by Subham Jain, available on [Kaggle](https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior). The dataset contains information provided by customers of a financial institution during the loan application process. 

Dataset Statistics:
- Dataset size: 252,000 records 
- Target variable: Risk flag (12.3% defaults)
- Features: 11 
  - Demographic: Age, married, profession
  - Financial: Income, house ownership, car ownership
  - Location: City, state
  - Behavioral: Experience, current job years, current house years

### Training Procedure
#### Preprocessing
The preprocessing of the raw data includes the following steps:
- Handling duplicates, data types, missing values, and outliers.
- Engineering new features: Job stability, city tier, and state default rate.
- Applying `StandardScaler` to numerical features, `OneHotEncoder` to nominal features, and `OrdinalEncoder` to ordinal features.

#### Training Hyperparameters
The final Random Forest Classifier model was trained with the following hyperparameters, identified through randomized search with 5-fold cross-validation:
- `n_estimators=225`  
- `max_depth=26`  
- `min_samples_split=2`  
- `min_samples_leaf=1`  
- `max_features=0.13`  
- `class_weight='balanced'` 

---

## Evaluation
The model was evaluated on a hold-out test set (10% of the data). The primary evaluation metric was the Area Under the Precision-Recall Curve (AUC-PR), which is well-suited for imbalanced datasets, where the focus is on the minority class (default). The decision threshold was optimized for the F1-score while ensuring minimum recall (≥0.75) and precision (≥0.50) for the default class. The final Random Forest model achieved an AUC-PR of 0.59 on the test set. 

**Classification Report (Test)**
|                        | Precision | Recall | F1-Score | Samples |
|------------------------|-----------|--------|----------|---------|
| Class 0: Non-Defaulter | 0.97      | 0.90   | 0.93     | 22122   |
| Class 1: Defaulter     | 0.51      | 0.79   | 0.62     | 3078    |
| Accuracy           |           |        | 0.88 | 25200   |
| Weighted Avg       | 0.91      | 0.88   | 0.89 | 25200   |

<img src="images/rf_confusion_matrix_test.png" alt="Final Random Forest: Confusion Matrix (Test)" width="500">

**Feature Importance**  
The most influential features in the model's predictions are income, age, and the engineered state default rate.  
![Feature Importance](images/rf_feature_importance_final.png)

---

## License
The model pipeline is licensed under [Apache-2.0](LICENSE). The source code of this project, hosted on [GitHub](https://github.com/JensBender/loan-default-prediction), and the source code of the web app hosted on [Hugging Face Spaces](https://huggingface.co/spaces/JensBender/loan-default-prediction-app), are licensed under the MIT License. 

---

## Citation
If you use this model in your work, please cite it as follows:
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
```