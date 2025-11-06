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
This model repository contains a `scikit-learn` pipeline for predicting loan defaults. The pipeline includes all data preprocessing steps and a Random Forest Classifier model trained on a dataset of 252,000 loan applications. The model predicts the probability of loan applicants defaulting on their loan based on information from loan application forms. It is designed to assist financial institutions in making more informed, data-driven lending decisions and managing credit risk. 

## Model Details
### Model Description
The model pipeline takes raw loan application data as a `pandas DataFrame` input and performs all necessary preprocessing steps such as feature engineering, scaling, and encoding. The pipeline then uses a Random Forest Classifier model to predict the probability of loan default as a `NumPy array` output. 

| Model Pipeline | Version | Framework | Task | Input | Output | Author | License |
|---------------|---------|-----------|------|-------|--------|--------|---------|
| Random Forest with preprocessing | 1.0 | Python, scikit-learn | Binary classification | Tabular data | Predicted probabilities | Jens Bender | Apache 2.0 |

### Model Sources
| Component | Link |
|----------|------|
| Source Code | [GitHub](https://github.com/JensBender/loan-default-prediction) |
| Model Pipeline | [Hugging Face Hub](https://huggingface.co/JensBender/loan-default-prediction-pipeline) |
| Web App | [Hugging Face Spaces](https://huggingface.co/spaces/JensBender/loan-default-prediction-app) |

### How to Use
The pipeline is serialized as a `joblib` file. You can load and use it for inference as shown below:

```python
import joblib
import pandas as pd

# Load the pipeline from the Hub
pipeline = joblib.load(hf_hub_download("JensBender/loan-default-prediction-pipeline", "pipeline.joblib"))

# Create a sample DataFrame with new data
# Note: The column names and data types must match the training data
new_data = pd.DataFrame({
    'income': [5000000],
    'age': [35],
    'experience': [10],
    'married': ['single'],
    'house_ownership': ['rented'],
    'car_ownership': ['no'],
    'profession': ['Software_Developer'],
    'city': ['Bangalore'],
    'state': ['Karnataka'],
    'current_job_years': [5],
    'current_house_years': [12]
})

# Get predicted probabilities
# Note: Returns np.ndarray with probabilities for both classes (0: no default, 1: default)
probabilities = pipeline.predict_proba(new_data)
default_probability = probabilities[0][1]

print(f"Probability of default: {default_probability:.2f}")

# Apply a custom threshold to make a classification decision
threshold = 0.29
prediction = "default" if default_probability >= threshold else "no default"
print(f"Prediction with threshold {threshold}: {prediction}")
```

---

## Uses
### Direct Use
The model is intended to be used as a tool to support credit risk assessment. It can be integrated into decision-making workflows to provide a quantitative measure of default risk for a loan applicant.

### Out-of-Scope Use
This model is **not** intended for:
- Fully automated lending decisions without human oversight. The model's predictions should not be the sole factor in any financial decision.
- Evaluating applicants from demographics, geographies, or economic environments not represented in the training data.
- Use in a production environment without rigorous, ongoing validation and fairness audits. 

---

## Bias, Risks, and Limitations
The model was trained on historical data, which may carry inherent biases related to socioeconomic status, geography, or other demographic factors. Consequently, the model may produce predictions that unfairly disadvantage certain groups of applicants.

### Recommendations
- **Human in the Loop:** Always use this model as part of a broader decision-making framework that includes human oversight.
- **Fairness and Bias Audits:** Before deploying this model in a live environment, conduct thorough fairness and bias analyses to ensure it performs equally across different demographic groups.
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
The model was evaluated on a hold-out test set (10% of the data). The primary metric was AUC-PR, suitable for the imbalanced nature of the dataset. The decision threshold was optimized on a validation set to maximize the F1-score while meeting minimum recall (≥0.75) and precision (≥0.50) criteria.

### Test Set Performance
| Metric              | Value  |
|---------------------|--------|
| AUC-PR              | 0.59   |
| Recall (Class 1)    | 0.79   |
| Precision (Class 1) | 0.51   |
| F1-Score (Class 1)  | 0.62   |
| Accuracy            | 0.88   |

### Classification Report (Test Set)
|                        | Precision | Recall | F1-Score | Samples |
|------------------------|-----------|--------|----------|---------|
| Class 0: Non-Defaulter | 0.97      | 0.90   | 0.93     | 22122   |
| Class 1: Defaulter     | 0.51      | 0.79   | 0.62     | 3078    |
| **Accuracy**           |           |        | **0.88** | **25200**   |
| **Macro Avg**          | **0.74**  | **0.84**| **0.78** | **25200**   |
| **Weighted Avg**       | **0.91**  | **0.88**| **0.89** | **25200**   |

### Confusion Matrix (Test Set)
![Confusion Matrix](images/rf_confusion_matrix_test.png)

### Feature Importance
The most influential features in the model's predictions are income, age, and the engineered `state_default_rate`.  
![Feature Importance](images/rf_feature_importance_final.png)

---

## Deployment
The end-to-end `scikit-learn` pipeline, which includes all preprocessing steps and the final Random Forest model, is serialized using `joblib`. The optimized decision threshold of **0.29** is not part of the pipeline itself and should be applied in post-processing.

The pipeline is served via a Dockerized web application with a FastAPI backend and a Gradio frontend, which is hosted on [Hugging Face Spaces](https://huggingface.co/spaces/JensBender/loan-default-prediction-app).

---

## License
The model pipeline is licensed under the [Apache-2.0 License](https://huggingface.co/JensBender/loan-default-prediction-pipeline/resolve/main/LICENSE). The project's source code is available on [GitHub](https://github.com/JensBender/loan-default-prediction) under the [MIT License](https://github.com/JensBender/loan-default-prediction/blob/main/LICENSE). The web app is hosted on [Hugging Face Spaces](https://huggingface.co/spaces/JensBender/loan-default-prediction-app) under the MIT License.

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