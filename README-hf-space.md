---
title: Loan Default Prediction
subtitle: Submit customer application data to predict loan default
emoji: 💰
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
app_port_secondary: 8000
pinned: false
models:
  - JensBender/loan-default-prediction-pipeline
tags:
  - finance
  - credit-risk
  - loan-default
  - tabular-data
  - scikit-learn
  - random-forest
  - gradio
  - fastapi
  - docker
---

## 🏦 Loan Default Prediction App
A web app that predicts loan default based on customer application data. Powered by machine learning, trained on over 250,000 loan applications, and with a user-friendly interface to help financial institutions make data-driven lending decisions.

### How to Use
1.  **Enter  Information**: Fill in the form with the applicant's details such as age and income.
2.  **Click Predict**: The app will process your input and display the prediction ("Default" or "No Default") along with probabilities.
3.  **Interpret Results**: Combine prediction with human judgment and additional information. Predictions should not be used as the sole factor for loan decisions. 

### Links
| Component | Description | Link |
|------------|--------------|------|
| **Model Pipeline** | Preprocessing and Random Forest Classifier | [Hugging Face Hub](https://huggingface.co/JensBender/loan-default-prediction-pipeline) |
| **Source Code** | Training, evaluation, and deployment code | [GitHub](https://github.com/JensBender/loan-default-prediction) |
| **Web App** | Deployed interactive demo | [Hugging Face Space](https://huggingface.co/spaces/JensBender/loan-default-prediction-app) |

### License
The source code for this web app on Hugging Face Spaces and the source code of the overall project on [GitHub](https://github.com/JensBender/loan-default-prediction) is licensed under the [MIT License](LICENSE). The model pipeline is licensed under [Apache-2.0](https://huggingface.co/JensBender/loan-default-prediction-pipeline/resolve/main/LICENSE).
