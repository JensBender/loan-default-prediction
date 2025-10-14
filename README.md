<!-- anchor tag for back-to-top links -->
<a name="readme-top"></a>

<!-- HEADER IMAGE  -->
<img src="images/header-image.webp">

<!-- SHORT SUMMARY  -->
Developed a machine learning model for loan default prediction to enhance risk management for financial institutions. Engineered, scaled, and encoded features. Trained, tuned and evaluated eight models. Deployed the threshold-optimized Random Forest model (AUC-PR: 0.62, class-1 recall: 0.80) as a Dockerized web app with FastAPI backend and Gradio frontend.

---

<!-- TABLE OF CONTENTS -->
## 📋 Table of Contents
<ol>
  <li>
    <a href="#-summary">Summary</a>
    <ul>
      <li><a href="#️-built-with">Built With</a></li>
    </ul>
  </li>
  <li>
    <a href="#-motivation">Motivation</a>
  </li>
  <li>
    <a href="#️-data">Data</a>
  </li>
  <li>
    <a href="#-data-preprocessing">Data Preprocessing</a>
  </li>
  <li>
    <a href="#-exploratory-data-analysis-eda">Exploratory Data Analysis (EDA)</a>
  </li>
  <li>
    <a href="#️-modeling">Modeling</a>
    <ul>
      <li><a href="#-baseline-models">Baseline Models</a></li>      
      <li><a href="#️-hyperparameter-tuning">Hyperparameter Tuning</a></li>
      <li><a href="#-final-model">Final Model</a></li>
    </ul>
  </li>
  <li>
    <a href="#-deployment">Deployment</a>
  </li>
  <li>
    <a href="#-testing">Testing</a>
  </li>
  <li>
    <a href="#️-license">License</a>
  </li>
  <li>
    <a href="#-credits">Credits</a>
  </li>
  <li>
    <a href="#-appendix">Appendix</a>
    <ul>
      <li><a href="#descriptive-statistics-for-numerical-columns">Descriptive Statistics for Numerical Columns</a></li>      
      <li><a href="#numerical-distributions-histograms">Numerical Distributions (Histograms)</a></li>      
      <li><a href="#categorical-frequencies-bar-plots">Categorical Frequencies (Bar Plots)</a></li>      
      <li><a href="#numerical-numerical-relationships-scatter-plots">Numerical-Numerical Relationships (Scatter Plots)</a></li>      
      <li><a href="#numerical-categorical-relationships-bar-plots">Numerical-Categorical Relationships (Bar Plots)</a></li> 
      <li><a href="#categorical-categorical-relationships-grouped-bar-plots">Categorical-Categorical Relationships (Grouped Bar Plots)</a></li> 
      <li><a href="#model-comparison-tables">Model Comparison Tables</a></li>      
      <li><a href="#model-threshold-optimization">Model Threshold Optimization</a></li>
    </ul>
  </li>
</ol>


<!-- SUMMARY -->
## 🎯 Summary
Developed a machine learning solution to predict loan defaults using customer application data, empowering financial institutions to make data-driven lending decisions and minimize risk. This project involved:
- **Data Preprocessing**: Engineered new features (e.g., job stability, city tier, state default rate), handled duplicates, data types, missing values, and outliers, scaled numerical features, and encoded categorical features.
- **Exploratory Data Analysis**: Analyzed descriptive statistics and visualized distributions, correlations and relationships.
- **Modeling**: Trained and evaluated eight baseline models (e.g., Logistic Regression, Random Forest, XGBoost) and tuned hyperparameters. The final model, a Random Forest classifier with optimized decision threshold, achieved an AUC-PR of 0.62, class-1 recall of 0.80, and class-1 precision of 0.54. Visualized feature importances and showed model prediction examples.
- **Deployment**: Served the full machine learning pipeline (data preprocessing and final model) as a web app using FastAPI backend and Gradio frontend within a single Docker container.
- **Testing**: Implemented comprehensive unit, integration, and end-to-end tests to validate individual components, their interactions, and entire user journeys.

### 🛠️ Built With
- [![Python][Python-badge]][Python-url]
- [![Pandas][Pandas-badge]][Pandas-url]
- [![Matplotlib][Matplotlib-badge]][Matplotlib-url] 
- [![Seaborn][Seaborn-badge]][Seaborn-url]
- [![scikit-learn][scikit-learn-badge]][scikit-learn-url]
- [![FastAPI][FastAPI-badge]][FastAPI-url]
- [![Pydantic][Pydantic-badge]][Pydantic-url]
- [![Gradio][Gradio-badge]][Gradio-url]
- [![Docker][Docker-badge]][Docker-url]
- [![pytest][pytest-badge]][pytest-url]
- [![Selenium][Selenium-badge]][Selenium-url]
- [![Hugging Face][HuggingFace-badge]][HuggingFace-url]
- [![Visual Studio Code][VisualStudioCode-badge]][VisualStudioCode-url]
- [![Jupyter Notebook][JupyterNotebook-badge]][JupyterNotebook-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MOTIVATION -->
## 💡 Motivation
**Problem**  
Predicting loan defaults is a challenging task due to the multitude of influencing factors such as customers' demographic, financial, location, and behavioral attributes. Traditional default prediction models often oversimplify complex relationships between customer features and default risk. Machine learning offers enhanced predictive capability by capturing non-linear patterns and intricate dependencies in loan application data, enabling more accurate predictions of loan default risk.

**Objectives**  
- Develop a machine learning model to predict loan defaults using customer data from loan applications.
- Compare multiple models (e.g., Logistic Regression, Random Forest, XGBoost).
  - Primary evaluation metric: Area Under the Precision-Recall Curve (AUC-PR), as it suits class imbalance (12.3% defaults) with a focus on preventing defaults.
  - Secondary evaluation metrics: Class-1-specific recall, precision, and F1-score.
  - Success criteria: Minimum class-1 recall of 0.75 and class-1 precision of 0.50 on the test data.
- Identify key factors influencing loan default risk through feature importance analysis.

**Value Proposition**  
This project enables financial institutions to reduce loan default rates and make better and faster lending decisions by leveraging machine learning for automated and improved risk assessment. 

**Business Goals**  
- Reduce losses by 5M-10M INR within 12 months of model deployment by decreasing the loan default rate by 10%-20%.
- Decrease loan processing time by 25%-40% by automating risk assessment, leading to less time spent on manual evaluations.
- Ensure 100% compliance with regulatory requirements and fair lending practices.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- DATA -->
## 🗂️ Data
The dataset contains information provided by customers of a financial institution during the loan application process. It is sourced from the "Loan Prediction Based on Customer Behavior" dataset by Subham Jain, available on [Kaggle](https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior). Stored in `Training Data.csv`, it contains the features, target variable (`Risk Flag`), and `ID` column. 

Dataset Statistics:
- Dataset size: 252,000 records 
- Target variable: Risk flag (12.3% defaults)
- Features: 11 
  - Demographic: Age, married, profession
  - Financial: Income, house ownership, car ownership
  - Location: City, state
  - Behavioral: Experience, current job years, current house years

Data Overview Table:

| Column | Description | Storage Type | Semantic Type | Theoretical Range | Observed Range |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Risk Flag | Defaulted on loan (0: No, 1: Yes) | Integer | Categorical (Binary) | [0, 1] | [0, 1] |
| Income | Income of the applicant | Integer | Numerical | [0, ∞] | [10K, 10M] |
| Age | Age of the applicant (in years) | Integer | Numerical | [18, ∞] | [21, 79] |
| Experience | Work experience (in years) | Integer | Numerical | [0, ∞] | [0, 20] |
| Profession | Applicant's profession | String | Categorical (Nominal) | Any profession [e.g., "Architect", "Dentist"] | 51 unique professions |
| Married | Marital status | String | Categorical (Binary) | ["single", "married"] | ["single", "married"] |
| House Ownership | Applicant owns or rents a house | String | Categorical (Nominal) | ["rented", "owned", "norent_noown"] | ["rented", "owned", "norent_noown"] |
| Car Ownership | Whether applicant owns a car | String | Categorical (Binary) | ["yes", "no"] | ["yes", "no"] |
| Current Job Years | Years in the current job | Integer | Numerical | [0, ∞] | [0, 14] |
| Current House Years | Years in the current house | Integer | Numerical | [0, ∞] | [10, 14] |
| City | City of residence | String | Categorical (Nominal) | Any city [e.g., "Mumbai", "Bangalore"] | 317 unique cities |
| State | State of residence | String | Categorical (Nominal) | Any state [e.g., "Maharashtra", "Tamil_Nadu"] | 29 unique states |

Example Data:

| Risk Flag | Income    | Age | Experience | Profession         | Married | House Ownership | Car Ownership | Current Job Years | Current House Years | City      | State         |
| :-------- | :-------- | :-- | :--------- | :----------------- | :------ | :-------------- | :------------ | :---------------- | :------------------ | :-------- | :------------ |
| 0         | 1,303,834 | 23  | 3          | Mechanical_engineer | single  | rented          | no            | 3                 | 13                   | Rewa      | Madhya_Pradesh |
| 1         | 6,256,451 | 41  | 2          | Software_Developer | single  | rented          | yes           | 2                 | 12                   | Bangalore | Tamil_Nadu    |
| 0         | 3,991,815 | 66  | 4          | Technical_writer   | married | rented          | no            | 4                 | 10                   | Alappuzha | Kerala        |

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- DATA PREPROCESSING -->
## 🧹 Data Preprocessing
Used `pandas` and `sklearn` for data loading, cleaning, transformation, and saving.
- **Loaded data** from .csv file using `pandas` `read_csv`.
- **Standardized column names and labels** to `snake_case` using `pandas` string methods and `apply` with custom functions.
- **Handled duplicates**: Verified the absence of duplicates using both the ID column and complete row comparison.
- **Handled data types**: Converted string columns with two categories to boolean columns using `pandas` `map`.
- **Train-validation-test split**: Split data into training (80%), validation (10%), and test (10%) sets using `sklearn` `train_test_split`.
- **Engineered new features**: Derived job stability from profession and city tier from city using mapping functions with  `pandas` `map`. Derived state default rate from state using target encoding.
- **Defined semantic type** for each column (numerical, categorical, boolean).
- **Handled missing values**: Verified the absence of missing values in all columns and datasets.
- **Handled outliers**: Identified multivariate outliers using `sklearn` `IsolationForest` and univariate outliers using statistical methods (3SD and 1.5 IQR) with custom transformer classes that inherit from `sklearn` `BaseEstimator` and `TransformerMixin`.
- **Feature scaling and encoding**:
    - Scaled numerical features: Used standard scaling with `sklearn` `StandardScaler`.
    - Encoded categorical features: Used one-hot encoding for nominal features (`sklearn` `OneHotEncoder`) and ordinal encoding for ordinal features (`OrdinalEncoder`).
    - Applied scaling and encoding together using `sklearn` `ColumnTransformer`.
- **Saved the preprocessed data** for training, validation, and test sets as .csv files using `pandas` `to_csv`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- EXPLORATORY DATA ANALYSIS -->
## 🔍 Exploratory Data Analysis (EDA)
Used `pandas`, `numpy`, `seaborn`, and `matplotlib` for statistical analysis and visualizations.
- **Univariate EDA**:
    - **Numerical columns**:
        - Analyzed [descriptive statistics](#descriptive-statistics-for-numerical-columns) (e.g., mean, median) using `pandas` `describe`.
        - Visualized distributions with [histograms](#numerical-distributions-histograms) using `seaborn` `histplot` and `matplotlib`.
    - **Categorical columns**:
        - Examined frequencies using `pandas` `value_counts`.
        - Visualized frequency distributions with [bar plots](#categorical-frequencies-bar-plots) using `seaborn` `barplot` and `matplotlib`. 
- **Bivariate EDA**:
    - **Numerical vs. numerical**:
        - Analyzed pairwise relationships with a correlation matrix (`pandas` `corr` and `numpy`) and visualized them with a heatmap (`seaborn` `heatmap`), see below.
        - Visualized relationships with [scatterplots](#numerical-numerical-relationships-scatter-plots) using `seaborn` `scatterplot` and `matplotlib`.
    - **Numerical vs. categorical**:
        - Explored relationships with group-wise statistics (e.g., mean or median by category) using `pandas` `groupby` and `agg`.
        - Quantified the magnitude of group differences with Cohen's d using a custom function.
        - Visualized results with [bar plots](#numerical-categorical-relationships-bar-plots) using `seaborn` `barplot` and `matplotlib`.
    - **Categorical vs. categorical**:
        - Analyzed relationships with contingency tables using `pandas` `crosstab`.
        - Visualized relationships with [grouped bar plots](#categorical-categorical-relationships-grouped-bar-plots) using `pandas` `crosstab` `plot` and `matplotlib`.

![Correlation Heatmap](images/correlation_heatmap.png)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Modeling -->
## 🏗️ Modeling
Trained, evaluated, and optimized multiple models using `sklearn` and `xgboost`.

### 🧱 Baseline Models  
- Trained eight baseline models with default hyperparameter values.
- Trained each model with four outlier handling methods. 
- Evaluated model performance using AUC-PR (primary metric) and class-1 recall, precision, and F1-score (secondary metrics). 

![AUC-PR Comparison by Baseline Model and Outlier Handling Method](images/aucpr_comparison_baseline.png)
Proceeded without outlier handling, as it did not meaningfully improve AUC-PR. Plotted precision-recall curves using `matplotlib`.

![Precision-Recall Curves: Baseline Models (No Outlier Handling)](images/precision_recall_curves_baseline.png)
The top-performing models were selected for hyperparameter tuning: Random Forest (AUC-PR: 0.62), XGBoost (0.56), K-Nearest Neighbors (0.56), and Decision Tree (0.47). See detailed metrics in [Appendix: Model Comparison Tables](#model-comparison-tables).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### ⚙️ Hyperparameter Tuning  
Performed random search with 5-fold cross-validation using `sklearn` `RandomizedSearchCV`. Evaluated the best-performing model from each algorithm using precision-recall curves on the validation data. Random Forest demonstrated the highest AUC-PR (0.62), followed by XGBoost (0.61).  
![Precision-Recall Curves: Hyperparameter-Tuned Models](images/precision_recall_curves_tuned.png)

**Threshold Optimization**  
Optimized decision thresholds of all tuned models to balance recall and precision based on business needs. For loan defaults, recall is often prioritized because missing a defaulter (a false negative) is costly. Determined the optimal thresholds by maximizing the F1-score while ensuring recall ≥ 0.80 and precision ≥ 0.40. The plot below shows threshold optimization for Random Forest. For all model plots, see [Appendix: Model Threshold Optimization](#model-threshold-optimization).
![Tuned Random Forest: Class-1 Metrics by Threshold](images/rf_metrics_by_threshold_tuned.png)

Compared performance of hyperparameter-tuned models with optimized decision thresholds on the validation data. Random Forest and XGBoost demonstrated the highest F1-score (0.64) while meeting minimum recall (0.80) and exceeding precision (0.54 vs. min. 0.40).
| Model               | AUC-PR | Recall (Class 1) | Precision (Class 1) | F1-Score (Class 1) | Accuracy |
|---------------------|--------|------------------|---------------------|-------------------|----------|
| K-Nearest Neighbors | 0.59   | 0.81             | 0.52                | 0.63              | 0.88     |
| Decision Tree       | 0.52   | 0.81             | 0.49                | 0.61              | 0.87     |
| Random Forest       | 0.62   | 0.80             | 0.54                | 0.64              | 0.89     |
| XGBoost             | 0.61   | 0.80             | 0.54                | 0.64              | 0.89     |

Diagnosed overfitting of optimized threshold models by comparing training and validation performance. Random Forest showed the lowest AUC-PR difference (0.06) compared to XGBoost (0.13), Decision Tree (0.13), and KNN (0.26).
<img src="images/overfitting_tuned_thresholds.png" alt="Tuned Threshold Models Overfitting: Train vs. Validation AUC-PR" width="600">

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### 🏆 Final Model
**Model Selection**  
Random Forest with optimized threshold of 0.29 was selected for its good performance, low overfitting, and interpretability.
- Performance: Highest AUC-PR (0.62) and F1-score (0.64, tied with XGBoost), meeting minimum recall and precision.
- Overfitting: Lowest AUC-PR difference between training and validation (0.06).  
- Interpretability: Higher degree of interpretability than XGBoost, crucial for transparency and regulatory compliance in finance.  
- Hyperparameters:
  - `n_estimators=225`
  - `max_depth=26`
  - `min_samples_split=2`
  - `min_samples_leaf=1`
  - `max_features=0.13`
  - `class_weight="balanced"`

**Model Evaluation**  
Compared metrics between training, validation, and test data. The performance drop from validation to test was small, suggesting consistent model behavior on unseen data.
| Data       | AUC-PR | Recall (Class 1) | Precision (Class 1) | F1-Score (Class 1) | Accuracy |
|------------|--------|--------------------|---------------------|--------------------|----------|
| Training   | 0.68   | 1.00               | 0.62                | 0.77               | 0.93     |
| Validation | 0.62   | 0.80               | 0.54                | 0.64               | 0.89     |
| Test       | 0.59   | 0.79               | 0.51                | 0.62               | 0.88     |

Evaluated model performance on the test data using a classification report and a confusion matrix. The model met the project-defined success criteria, achieving recall ≥ 0.75 and precision ≥ 0.50 for defaulters on unseen data.
|                        | Precision | Recall | F1-Score | Samples |
|------------------------|-----------|--------|----------|---------|
| Class 0: Non-Defaulter | 0.97      | 0.90   | 0.93     | 22122   |
| Class 1: Defaulter     | 0.51      | 0.79   | 0.62     | 3078    |
| Accuracy               |           |        | 0.88     | 25200   |
| Macro Avg              | 0.74      | 0.84   | 0.78     | 25200   |
| Weighted Avg           | 0.91      | 0.88   | 0.89     | 25200   |

<img src="images/rf_confusion_matrix_test.png" alt="Final Random Forest: Confusion Matrix (Test)" width="500">

**Feature Importance**  
The most influential features were income, age, and state default rate, indicating that financial stability and regional risk factors heavily influence loan defaults. Other factors like experience and current job years also contributed moderately, while demographic factors like car ownership and marital status had minimal impact.
![Final Random Forest: Feature Importance](images/rf_feature_importance_final.png)

**Model Prediction Examples**  
The table below illustrates model performance on test data by highlighting best examples (correct, high confidence), worst examples (incorrect, high confidence), and typical examples (average confidence). The model demonstrated reliable predictions on typical cases but displayed problematic overconfidence on misclassified edge cases.

| Example | Income    | Age | State Default Rate | Experience | Actual Default | Predicted Default | Confidence | Misclassified |
| ------- | --------- | --- | ------------------ | ---------- | -------------- | ----------------- | ---------- | ------------- |
| Best    | 495,619   | 26  | 12.8%              | 1          | Yes            | Yes               | 99%        | ✅ No         |
| Best    | 2,901,323 | 56  | 13.7%              | 2          | No             | No                | 100%       | ✅ No         |
| Worst   | 8,290,834 | 42  | 12.8%              | 3          | Yes            | No                | 95%        | ❌ Yes        |
| Worst   | 7,644,982 | 24  | 12.2%              | 1          | No             | Yes               | 98%        | ❌ Yes        |
| Typical | 4,570,845 | 47  | 15.5%              | 3          | Yes            | Yes               | 94%        | ✅ No         |
| Typical | 8,391,288 | 24  | 11.7%              | 4          | No             | No                | 94%        | ✅ No         |

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- DEPLOYMENT -->
## 🚀 Deployment
Built a machine learning (ML) pipeline including the final Random Forest model and all data preprocessing steps using `scikit-learn` and saved it to file with `joblib`. Deployed the loan default prediction pipeline as a web application using `FastAPI`, `Gradio`, and `Docker`. 

**FastAPI Backend**  
- Served the ML pipeline via a RESTful API using `FastAPI` that handles single and batch requests.
- Implemented data validation using `Pydantic` models with required and optional fields, type constraints, enums for categorical fields, value range constraints for numerical fields, and custom validation logic. 
- The `/predict` endpoint accepts loan application data in JSON format, processes it through the ML pipeline, and returns the prediction ("default" or "no default") along with probability scores.

**Gradio Frontend**   
- Built a user-friendly interface with `Gradio` so users can enter applicant information through a simple form.  
- Sends submissions to the FastAPI backend, which returns predictions that are rendered in the UI.

**Docker**    
- Containerized the application with `Docker` for reliable and portable deployment. 
- Served both the `FastAPI` backend and `Gradio` frontend together from a single `Docker` container.
- The provided `Dockerfile` defines the Python environment, installs dependencies, and launches backend and frontend with the `start.sh` shell script.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- TESTING -->
## 🕵 Testing
Utilized `pytest` for unit, integration, and end-to-end testing.
- **Unit Tests** (`tests/unit/`):  Verified individual components in isolation. 
  - Custom transformers (`src/custom_transformers.py`): Ensured correct data transformation, scikit-learn compatibility, and error handling.
  - Pydantic schemas (`backend/schemas.py`): Validated API data models for type constraints, optional fields, and error responses.
  - Helper functions (`backend/app.py`, `frontend/app.py`): Tested utility functions for backend and frontend logic.
- **Integration Tests** (`tests/integration/`): Ensured different parts of the application work together.  
  - ML Pipelines (`src/pipeline.py`): Tested full and partial machine learning pipelines.
  - FastAPI backend: Verified `/predict` endpoint behavior (validation, batch processing, error handling).
  - Frontend-backend: Tested communication between frontend UI and backend API.
- **End-to-End Tests** (`tests/e2e/`): Simulated user journeys in a browser with `Selenium`.  
  - UI interaction: Automated filling and submitting of the loan application form.
  - Happy/error paths: Covered both valid and invalid inputs to ensure correct predictions and error messages in the UI.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## ©️ License
This project is licensed under the [MIT License](LICENSE).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CREDITS -->
## 👏 Credits
This project was made possible with the help of the following resources:
- **Header image**: Generated using the FLUX.1 [dev] image generator via [Hugging Face](https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev) by [Black Forest Labs](https://blackforestlabs.ai/).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- APPENDIX -->
## 📎 Appendix
### Descriptive Statistics for Numerical Columns
Training Data
| Column              | Count  | Mean          | Std            | Min      | 25%          | 50%          | 75%          | Max        |
|---------------------|--------|---------------|----------------|----------|--------------|--------------|--------------|------------|
| Income              | 201,600| 5,000,741  | 2,880,711   | 10,310| 2,499,018 | 5,004,535 | 7,489,827 | 9,999,938|
| Age                 | 201,600| 50.01         | 17.08          | 21.00    | 35.00        | 50.00        | 65.00        | 79.00      |
| Experience          | 201,600| 10.09         | 6.00           | 0.00     | 5.00         | 10.00        | 15.00        | 20.00      |
| Current Job Yrs     | 201,600| 6.33          | 3.65           | 0.00     | 3.00         | 6.00         | 9.00         | 14.00      |
| Current House Yrs   | 201,600| 12.00         | 1.40           | 10.00    | 11.00        | 12.00        | 13.00        | 14.00      |
| State Default Rate  | 201,600| 0.12          | 0.02           | 0.05     | 0.11         | 0.12         | 0.13         | 0.21       |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Numerical Distributions (Histograms)
![Numerical Distributions (Histogram Matrix)](images/numerical_distributions_histograms.png)
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Categorical Frequencies (Bar Plots)
![Categorical Frequencies (Bar Plot Matrix)](images/categorical_frequencies_barplots.png)
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Numerical-Numerical Relationships (Scatter Plots)
![Numerical-Numerical Relationships (Scatter Plot Matrix)](images/numerical_relationships_scatterplots.png)
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Numerical-Categorical Relationships (Bar Plots)
![Numerical-Categorical Relationships (Bar Plot Matrix)](images/numerical_categorical_relationships_barplots.png)
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Categorical-Categorical Relationships (Grouped Bar Plots)
![Categorical-Categorical Relationships (Grouped Bar Plot Matrix)](images/categorical_relationships_groupedbarplots.png)
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Model Comparison Tables
**Baseline Models (No Outlier Handling)**
| Model                  | AUC-PR | Recall (Class 1) | Precision (Class 1) | F1-Score (Class 1) | Accuracy |
|------------------------|--------|------------------|----------------------|--------------------|----------|
| Logistic Regression    | 0.16   | 0.00             | 0.00                 | 0.00               | 0.88     |
| Elastic Net            | 0.16   | 0.00             | 0.00                 | 0.00               | 0.88     |
| K-Nearest Neighbors    | 0.53   | 0.50             | 0.56                 | 0.53               | 0.89     |
| Support Vector Machine | 0.13   | 0.28             | 0.13                 | 0.18               | 0.67     |
| Neural Network         | 0.32   | 0.10             | 0.51                 | 0.17               | 0.88     |
| Decision Tree          | 0.46   | 0.57             | 0.52                 | 0.55               | 0.88     |
| Random Forest          | 0.60   | 0.53             | 0.60                 | 0.56               | 0.90     |
| XGBoost                | 0.54   | 0.20             | 0.64                 | 0.30               | 0.89     |

**Hyperparameter-Tuned Models (Default Thresholds)**
| Model               | AUC-PR | Recall (Class 1) | Precision (Class 1) | F1-Score (Class 1) | Accuracy |
|---------------------|--------|------------------|---------------------|-------------------|----------|
| K-Nearest Neighbors | 0.59   | 0.54             | 0.59                | 0.56              | 0.89     |
| Decision Tree       | 0.52   | 0.87             | 0.46                | 0.60              | 0.86     |
| Random Forest       | 0.62   | 0.77             | 0.57                | 0.65              | 0.90     |
| XGBoost             | 0.61   | 0.81             | 0.53                | 0.64              | 0.89     |

**Hyperparameter-Tuned Models (Optimized Thresholds)**
| Model               | AUC-PR | Recall (Class 1) | Precision (Class 1) | F1-Score (Class 1) | Accuracy |
|---------------------|--------|------------------|---------------------|-------------------|----------|
| K-Nearest Neighbors | 0.59   | 0.81             | 0.52                | 0.63              | 0.88     |
| Decision Tree       | 0.52   | 0.81             | 0.49                | 0.61              | 0.87     |
| Random Forest       | 0.62   | 0.80             | 0.54                | 0.64              | 0.89     |
| XGBoost             | 0.61   | 0.80             | 0.54                | 0.64              | 0.89     |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Model Threshold Optimization
**Random Forest**
![Tuned Random Forest: Class-1 Metrics by Threshold](images/rf_metrics_by_threshold_tuned.png)

**XGBoost**
![Tuned XGBoost: Class-1 Metrics by Threshold](images/xgb_metrics_by_threshold_tuned.png)

**Decision Tree**
![Tuned Decision Tree: Class-1 Metrics by Threshold](images/tree_metrics_by_threshold_tuned.png)

**K-Nearest Neighbors**
![Tuned KNN: Class-1 Metrics by Threshold](images/knn_metrics_by_threshold_tuned.png)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS -->
[Python-badge]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[Pandas-badge]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org/
[Matplotlib-badge]: https://img.shields.io/badge/Matplotlib-%23f0f0f0.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[Matplotlib-url]: https://matplotlib.org/
[Seaborn-badge]: https://img.shields.io/badge/seaborn-%230C4A89.svg?style=for-the-badge&logo=seaborn&logoColor=white
[Seaborn-url]: https://seaborn.pydata.org/
[scikit-learn-badge]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[scikit-learn-url]: https://scikit-learn.org/stable/
[FastAPI-badge]: https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white
[FastAPI-url]: https://fastapi.tiangolo.com/
[Pydantic-badge]: https://img.shields.io/badge/Pydantic-3776AB?style=for-the-badge&logo=pydantic&logoColor=white
[Pydantic-url]: https://docs.pydantic.dev/
[Gradio-badge]: https://img.shields.io/badge/Gradio-fc7404?style=for-the-badge&logo=gradio&logoColor=white
[Gradio-url]: https://gradio.app
[Docker-badge]: https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white
[Docker-url]: https://www.docker.com/
[pytest-badge]: https://img.shields.io/badge/pytest-00A9E5?style=for-the-badge&logo=pytest&logoColor=white
[pytest-url]: https://docs.pytest.org/
[Selenium-badge]: https://img.shields.io/badge/selenium-%43B02A.svg?style=for-the-badge&logo=selenium&logoColor=white
[Selenium-url]: https://www.selenium.dev/
[HuggingFace-badge]: https://img.shields.io/badge/Hugging%20Face-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black
[HuggingFace-url]: https://huggingface.co/
[VisualStudioCode-badge]: https://img.shields.io/badge/Visual%20Studio%20Code-0078D4?style=for-the-badge&logo=visualstudiocode&logoColor=white
[VisualStudioCode-url]: https://code.visualstudio.com/
[JupyterNotebook-badge]: https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white
[JupyterNotebook-url]: https://jupyter.org/
