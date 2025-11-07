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
---

## 🏦 Loan Default Prediction App
This web application predicts loan default based on customer application data. It provides a user-friendly interface to interact with the underlying prediction model, which is powered by machine learning and was trained on over 250,000 loan applications.

### How to Use
1.  **Enter  Information**: Fill in the form with the applicant's age, income, and other information.
2.  **Click Predict**: Click the "Predict" button below the form.
3.  **View Results**: The app will return a prediction ("Default" or "No Default") along with probability scores.

### Links
| Component | Link |
|---|---|
| **Model Pipeline** | [Hugging Face Hub](https://huggingface.co/JensBender/loan-default-prediction-pipeline) |
| **Source Code** | [GitHub](https://github.com/JensBender/loan-default-prediction) |

### License
The source code for this web app is licensed under the [MIT License](https://github.com/JensBender/loan-default-prediction/blob/main/LICENSE). The model pipeline is licensed under [Apache-2.0](https://huggingface.co/JensBender/loan-default-prediction-pipeline/resolve/main/LICENSE).
