---
title: Loan Default Prediction
subtitle: Submit customer application data to predict loan default
emoji: 💰
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
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
A web application that predicts loan default based on customer application data, helping financial institutions make data-driven lending decisions.  
Built with `Gradio`, `FastAPI`, and a `scikit-learn` Random Forest model trained on over 250,000 loan applications.

---

### How to Use 
1.  **Fill in Form**: Enter applicant details such as age, income, and experience.
2.  **Click Predict**: The app will process your input and return a "Default" or "No Default" prediction along with probabilities.
3.  **Interpret Responsibly**: Use the prediction as decision support, not as the sole basis for loan approval.  

### Use via API
You can also send requests directly to the FastAPI backend for programmatic access. This is useful for integrating the model into other applications or systems.

Example API usage with Python's `requests` library:
```python
import requests 

# Create example applicant data (JSON payload)
applicant_data = {
    "income": 300000,
    "age": 30,
    "experience": 3,
    "married": "single",
    "house_ownership": "rented",
    "car_ownership": "no",
    "profession": "artist",
    "city": "sikar",
    "state": "rajasthan",
    "current_job_yrs": 3,
    "current_house_yrs": 11,
}

# API request to FastAPI predict endpoint on Hugging Face Spaces
prediction_api_url = "https://jensbender-loan-default-prediction-app.hf.space/api/predict"
response = requests.post(prediction_api_url, json=applicant_data)

# Check if request was successful
response.raise_for_status()

# Extract prediction and probability of default
prediction_response = response.json()
prediction_result = prediction_response["results"][0]
prediction = prediction_result["prediction"]
default_probability = prediction_result["probabilities"]["Default"]

# Show results
print(f"Probability of default: {default_probability * 100:.1f}% (threshold: 29.0%)")
print(f"Prediction: {prediction}")
```

---

### How It Works
1. **Gradio Frontend (UI Layer)**  
    - Provides a clean and simple form for data entry.  
    - Sends form data as JSON to the backend API.  
    - Displays prediction results and probabilities in real time.
2. **FastAPI Backend (API Layer)**  
    - Receives requests from the `Gradio` frontend or direct API calls.  
    - Loads the pre-trained pipeline from the [Hugging Face Hub](https://huggingface.co/JensBender/loan-default-prediction-pipeline).  
    - Validates and passes data through the pipeline, and applies the decision threshold.  
    - Returns JSON responses containing predictions and probabilities.
3. **ML Pipeline (Model Layer)**  
   - Implements a full `scikit-learn` pipeline with preprocessing and a Random Forest Classifier model.  
   - Performs feature engineering, scaling, and encoding.  
   - Outputs predicted probabilities for both classes ("Default" and "No Default").
4. **Deployment Environment**
    - Packaged as a single `Docker` container.  
    - Runs seamlessly on Hugging Face Spaces using the Docker SDK.  

---

### Resources
| Component | Description | Link |
|------------|--------------|------|
| **Source Code** | Full project repository with training, evaluation, and deployment scripts | [GitHub](https://github.com/JensBender/loan-default-prediction) |
| **Model Pipeline** | Pre-trained `scikit-learn` pipeline with Random Forest Classifier and preprocessing | [Hugging Face Hub](https://huggingface.co/JensBender/loan-default-prediction-pipeline) |
| **Web App** | Live, interactive demo with Gradio frontend and FastAPI backend | [Hugging Face Spaces](https://huggingface.co/spaces/JensBender/loan-default-prediction-app) |

---

### Responsible Use
The model and by extension the web app is intended as a decision-support tool to assist in credit risk evaluation. It should **not** be used for automated lending decisions without human oversight.

---

### License
The source code for this web app on Hugging Face Spaces and the source code of the overall project on [GitHub](https://github.com/JensBender/loan-default-prediction) is licensed under the [MIT License](LICENSE). The model pipeline is licensed under [Apache-2.0](https://huggingface.co/JensBender/loan-default-prediction-pipeline/resolve/main/LICENSE).
