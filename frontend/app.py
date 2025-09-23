# --- Imports ---
# Standard library imports
import re
from typing import Any

# Third-party library imports
import gradio as gr
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException

# Local imports
from src.global_constants import (
    MARRIED_LABELS,
    CAR_OWNERSHIP_LABELS,
    HOUSE_OWNERSHIP_LABELS,
    PROFESSION_LABELS,
    CITY_LABELS,
    STATE_LABELS
)

# --- Constants ---
# Backend URL to FastAPI predict endpoint
BACKEND_URL = "http://127.0.0.1:8000/predict"

# Format categorical string labels (snake_case) for display in UI
MARRIED_DISPLAY_LABELS = [label.title() for label in MARRIED_LABELS]
CAR_OWNERSHIP_DISPLAY_LABELS = [label.title() for label in CAR_OWNERSHIP_LABELS]
HOUSE_OWNERSHIP_DISPLAY_LABELS = [label.replace("norent_noown", "Neither Rented Nor Owned").title() for label in HOUSE_OWNERSHIP_LABELS]
PROFESSION_DISPLAY_LABELS = [label.replace("_", " ").title() for label in PROFESSION_LABELS]
CITY_DISPLAY_LABELS = [label.replace("_", " ").title() for label in CITY_LABELS]
STATE_DISPLAY_LABELS = [label.replace("_", " ").title() for label in STATE_LABELS]


# --- Input Preprocessing Functions ---
# Format a string in snake_case (return non-string unchanged)
def format_snake_case(value: Any) -> Any:
    if isinstance(value, str):
        # Remove leading/trailing whitespace, convert to lowercase, and replace single or multiple hyphens, forward slashes, and inner whitespaces with a single underscore
        return re.sub(r"[-/\s]+", "_", value.strip().lower())
    return value  # return non-string unchanged


# Format all string values in a dictionary in snake_case
def snake_case_str_values_in_dict(inputs: dict[str, Any]) -> dict[str, Any]:
    return {key: format_snake_case(value) for key, value in inputs.items()}


# Format "house_ownership" label as expected by API backend
def format_house_ownership(display_label: Any) -> Any:
    if isinstance(display_label, str):
        return display_label.replace("neither_rented_nor_owned", "norent_noown")
    return display_label  # return non-string unchanged


# --- Error Handling ---
# Function to format Pydantic validation error from FastAPI backend into a user-friendly message for Gradio frontend output
def _format_validation_error(error_detail: dict) -> str:
    if "detail" in error_detail and isinstance(error_detail["detail"], list):
        wrong_inputs = []
        for error in error_detail["detail"]:
            wrong_input = error["loc"]
            type = error["type"]
            wrong_input_msg = f"{wrong_input} should be a valid {type}"
            wrong_inputs.append(wrong_input_msg)
    return "Please correct the following:\n" + "\n".join(wrong_inputs)

# --- Function to Predict Loan Default for Gradio UI ---
def predict_loan_default(
    age: int | float, 
    married: str, 
    income: int | float, 
    car_ownership: str, 
    house_ownership: str, 
    current_house_yrs: int | float, 
    city: str, 
    state: str, 
    profession: str, 
    experience: int | float, 
    current_job_yrs: int | float 
) -> tuple[str, dict[str, float]] | tuple[str, str]:
    try:
        # TEMPORARY: Print types and values for all inputs returned by Gradio components
        print("=== GRADIO COMPONENT TYPES ===")
        for var_name, var_value in [
            ("age", age),
            ("married", married),
            ("income", income),
            ("car_ownership", car_ownership),
            ("house_ownership", house_ownership),
            ("current_house_yrs", current_house_yrs),
            ("city", city),
            ("state", state),
            ("profession", profession),
            ("experience", experience),
            ("current_job_yrs", current_job_yrs)
        ]:
            print(f"{var_name}: {type(var_value)} = {var_value}")
        print("=================================")
  
        # --- Input preprocessing ---
        # Create inputs dictionary 
        inputs = {
            "income": income, 
            "age": age,
            "experience": experience,
            "married": married,
            "house_ownership": house_ownership,
            "car_ownership": car_ownership,
            "profession": profession,
            "city": city,
            "state": state,
            "current_job_yrs": current_job_yrs,
            "current_house_yrs": current_house_yrs
        }

        # Format string values in snake_case
        inputs = snake_case_str_values_in_dict(inputs)

        # Format "house_ownership" label as expected by API backend 
        inputs["house_ownership"] = format_house_ownership(inputs["house_ownership"])
        
        # --- Predict loan default ---       
        # Get prediction via post request to FastAPI backend
        response = requests.post(BACKEND_URL, json=inputs) 
        # Handle HTTP errors
        if response.status_code == 422:
            error_detail = response.json()
            error_message = _format_validation_error(error_detail)
            return f"Input Error:\n{error_message}", f"{error_detail}" 
        # Get prediction and probabilities for Gradio output
        prediction_response = response.json()
        prediction_result = prediction_response["results"][0]
        prediction = prediction_result["prediction"]
        probabilities = prediction_result["probabilities"]
        
        return prediction, probabilities

    except ConnectionError:
        return "Connection Error", "Could not connect to the prediction service. Please ensure the backend is running and try again."
    except Timeout:
        return "Timeout Error", "The request to the prediction service timed out. The service may be busy or slow. Please try again later."
    except RequestException:  # catches other frontend-to-backend communication errors
        return "Error" "The prediction service is temporarily unavailable due to an internal communication issue. Please try again later."
    except Exception as e:
        return "Error", f"{str(e)}"


# --- Gradio App UI ---
# Custom CSS 
custom_css = """
.narrow-centered-column {
    max-width: 700px; 
    width: 100%; 
    margin: 0 auto; 
}
#predict-button-wrapper {
    max-width: 250px;
    margin: 0 auto;
}
#prediction-text textarea {font-size: 1.8em; font-weight: bold; text-align: center;}
#pred-proba-label {margin-top: -15px;}
#markdown-note {margin-top: -13px;}
"""

# Create Gradio app UI using Blocks
with gr.Blocks(css=custom_css) as app_ui:
    # Title and description
    gr.Markdown(
        """
        <h1 style='text-align:center'>Loan Default Prediction</h1>
        <p style='text-align:center'>Submit the customer application data to receive an automated loan default prediction powered by machine learning.</p>
        """
    )

    # Inputs
    with gr.Group():
        with gr.Row():
            age = gr.Number(label="Age", value="")
            married = gr.Dropdown(label="Married/Single", choices=MARRIED_DISPLAY_LABELS, value=None)
            income = gr.Number(label="Income", value="")
        with gr.Row():
            car_ownership = gr.Dropdown(label="Car Ownership", choices=CAR_OWNERSHIP_DISPLAY_LABELS, value=None)
            house_ownership = gr.Dropdown(label="House Ownership", choices=HOUSE_OWNERSHIP_DISPLAY_LABELS, value=None)
            current_house_yrs = gr.Slider(label="Current House Years", minimum=10, maximum=14, step=1)
        with gr.Row():
            city = gr.Dropdown(label="City", choices=CITY_DISPLAY_LABELS, value=None)
            state = gr.Dropdown(label="State", choices=STATE_DISPLAY_LABELS, value=None)
            profession = gr.Dropdown(label="Profession", choices=PROFESSION_DISPLAY_LABELS, value=None)
        with gr.Row():
            experience = gr.Slider(label="Experience", minimum=0, maximum=20, step=1)
            current_job_yrs = gr.Slider(label="Current Job Years", minimum=0, maximum=14, step=1)
            gr.Markdown("")  # empty space for layout

    # Predict button 
    with gr.Column(elem_id="predict-button-wrapper"):
        predict = gr.Button("Predict")
    
    # Outputs
    with gr.Column(elem_classes="narrow-centered-column"):
        prediction_text = gr.Textbox(placeholder="Prediction Result", show_label=False, container=False, elem_id="prediction-text")   
        pred_proba = gr.Label(show_label=False, show_heading=False, elem_id="pred-proba-label")
        gr.Markdown(
            "<small>Note: Prediction uses an optimized decision threshold of 0.29 "
            "(predicts 'Default' if probability ≥ 29%, otherwise 'No Default').</small>",
            elem_id="markdown-note"
        )

    # Predict button click event
    predict.click(
        predict_loan_default,
        inputs=[
            age, married, income, car_ownership, house_ownership, current_house_yrs, 
            city, state, profession, experience, current_job_yrs
        ],
        outputs=[prediction_text, pred_proba]
    )


# --- Launch Web App UI ---
if __name__ == "__main__":
    app_ui.launch()
