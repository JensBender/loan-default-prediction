# --- Imports ---
# Standard library imports
import re
import logging
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

# --- Logger ---
# Setup a named logger for the Gradio frontend (to distinguish logs from backend)
logger = logging.getLogger("gradio_frontend")
logger.setLevel(logging.INFO)
# Handler to write logs to the console
stream_handler = logging.StreamHandler()
# Formatter for the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
stream_handler.setFormatter(formatter)
# Add handler to logger
logger.addHandler(stream_handler)

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
# Map internal input field names (snake_case) to user-friendly error messages
field_to_error_map = {
    "age": "Age: Enter a number between 21 and 79.",
    "married": "Married/Single: Select 'Married' or 'Single'",
    "income": "Income: Enter a number that is 0 or greater.",
    "car_ownership": "Car Ownership: Select 'Yes' or 'No'.",
    "house_ownership": "House Ownership: Select 'Rented', 'Owned', or 'Neither Rented Nor Owned'.",
    "current_house_yrs": "Current House Years: Enter a number between 10 and 14.",
    "city": "City: Select a city from the list.",
    "state": "State: Select a state from the list.",
    "profession": "Profession: Select a profession from the list.",
    "experience": "Experience: Enter a number between 0 and 20.",
    "current_job_yrs": "Current Job Years: Enter a number between 0 and 14.",
}


# Function to format Pydantic validation error from FastAPI backend into a user-friendly message for Gradio frontend output
def _format_validation_error(error_detail: dict) -> str:
    if "detail" in error_detail and isinstance(error_detail["detail"], list):
        all_errors = error_detail["detail"]
        error_msg = "Input Error!\n"
        for field in field_to_error_map:
            if any(field in error["loc"] for error in all_errors):
                error_msg += f"{field_to_error_map.get(field)}\n"
    return error_msg

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
        response = requests.post(
            BACKEND_URL, 
            json=inputs, 
            timeout=(3, 60)  # 3s connect timeout, 60s read timeout (receive first byte of response)
        ) 

        # Handle HTTP errors
        if response.status_code == 422:
            error_detail = response.json()
            error_message = _format_validation_error(error_detail)
            return error_message, "" 

        # Raise error for other bad status codes (4xx or 5xx)
        response.raise_for_status()

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
        return "Communication Error", "There was a problem communicating with the prediction service. Please try again later."
    except (KeyError, IndexError):
        return "Prediction Response Error", "The prediction service returned a response with an invalid prediction format. Please ensure the prediction service is set up correctly."
    except Exception as e:
        return "Error", f"An unexpected error has occurred. Please verify your inputs or try again later."


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
