# Standard library imports
import os
import re
import pickle

# Third-party library imports
import gradio as gr
import pandas as pd

# Local imports
from app.custom_transformers import (
    MissingValueChecker, 
    MissingValueStandardizer, 
    RobustSimpleImputer,
    SnakeCaseFormatter, 
    BooleanColumnTransformer, 
    JobStabilityTransformer, 
    CityTierTransformer, 
    StateDefaultRateTargetEncoder,
    RobustStandardScaler,
    RobustOneHotEncoder,
    RobustOrdinalEncoder,
    FeatureSelector
)
from app.global_constants import (
    MARRIED_LABELS,
    CAR_OWNERSHIP_LABELS,
    HOUSE_OWNERSHIP_LABELS,
    PROFESSION_LABELS,
    CITY_LABELS,
    STATE_LABELS
)

# Format categorical string labels for display in UI
MARRIED_DISPLAY_LABELS = [label.title() for label in MARRIED_LABELS]
CAR_OWNERSHIP_DISPLAY_LABELS = [label.title() for label in CAR_OWNERSHIP_LABELS]
HOUSE_OWNERSHIP_DISPLAY_LABELS = [label.replace("norent_noown", "Neither Rented Nor Owned").title() for label in HOUSE_OWNERSHIP_LABELS]
PROFESSION_DISPLAY_LABELS = [label.replace("_", " ").title() for label in PROFESSION_LABELS]
CITY_DISPLAY_LABELS = [label.replace("_", " ").title() for label in CITY_LABELS]
STATE_DISPLAY_LABELS = [label.replace("_", " ").title() for label in STATE_LABELS]


# --- Functions: Input Preprocessing ---
# Format a single string input in snake_case  
def snake_case_format(value):
    if isinstance(value, str):
        # Remove leading/trailing whitespace, convert to lowercase, and replace single or multiple hyphens, forward slashes, and inner whitespaces with a single underscore
        return re.sub(r"[-/\s]+", "_", value.strip().lower())
    return value  # return non-string values unchanged


# Format all string inputs in a dictionary in snake_case
def snake_case_format_inputs(inputs_dict):
    return {key: snake_case_format(value) for key, value in inputs_dict.items()}


# Format "house_ownership" label as expected by pipeline
def format_house_ownership(display_label):
    if isinstance(display_label, str):
        return display_label.replace("neither_rented_nor_owned", "norent_noown")
    return display_label  # return non-string values unchanged


# Convert float to int (pipeline was trained on int inputs)
def convert_float_to_int(value):
    if isinstance(value, bool):
        raise TypeError()  # otherwise Python would treat True as 1 and False as 0 not raising a TypeError 
    return int(round(value)) 


# --- Functions: Input Validation ---
# Check missing values in the inputs dictionary
def check_missing_values(inputs_dict):
    missing_inputs = []
    if inputs_dict["age"] in [None, "", [], {}, ()]:
        missing_inputs.append("Age")
    if not inputs_dict["married"]:  # catches 0, 0.0, False, None, "", [], {}, ()
        missing_inputs.append("Married/Single")
    if inputs_dict["income"] in [None, "", [], {}, ()]:
        missing_inputs.append("Income")
    if not inputs_dict["car_ownership"]:
        missing_inputs.append("Car Ownership")
    if not inputs_dict["house_ownership"]:
        missing_inputs.append("House Ownership")
    if inputs_dict["current_house_yrs"] in [None, "", [], {}, ()]:
        missing_inputs.append("Current House Years")
    if not inputs_dict["city"]:
        missing_inputs.append("City")
    if not inputs_dict["state"]:
        missing_inputs.append("State")
    if not inputs_dict["profession"]:
        missing_inputs.append("Profession")
    if inputs_dict["experience"] in [None, "", [], {}, ()]:
        missing_inputs.append("Experience")
    if inputs_dict["current_job_yrs"] in [None, "", [], {}, ()]:
        missing_inputs.append("Current Job Years")
    if len(missing_inputs) == 1:
        return f"Please provide: {missing_inputs[0]}."
    if len(missing_inputs) > 1:
        return f"Please provide: {', '.join(missing_inputs[:-1])} and {missing_inputs[-1]}."
    return None  # no missing values


# Validate data types in the inputs dictionary
def validate_data_types(inputs_dict):
    invalid_numbers = []
    invalid_strings = []
    invalid_datatype_message = "Data type error! "   

    # Numerical inputs     
    if not isinstance(inputs_dict["age"], (int, float)) or isinstance(inputs_dict["age"], bool):
        invalid_numbers.append("Age")
    if not isinstance(inputs_dict["income"], (int, float)) or isinstance(inputs_dict["income"], bool):
        invalid_numbers.append("Income")
    if not isinstance(inputs_dict["current_house_yrs"], (int, float)) or isinstance(inputs_dict["current_house_yrs"], bool):
        invalid_numbers.append("Current House Years")
    if not isinstance(inputs_dict["experience"], (int, float)) or isinstance(inputs_dict["experience"], bool):
        invalid_numbers.append("Experience")
    if not isinstance(inputs_dict["current_job_yrs"], (int, float)) or isinstance(inputs_dict["current_job_yrs"], bool):
        invalid_numbers.append("Current Job Years")
    if len(invalid_numbers) == 1:
        invalid_datatype_message += f"{invalid_numbers[0]} must be a number."
    if len(invalid_numbers) > 1:
        invalid_datatype_message += f"{', '.join(invalid_numbers[:-1])} and {invalid_numbers[-1]} must be numbers."
    
    # String inputs
    if not isinstance(inputs_dict["married"], str):
        invalid_strings.append("Married/Single")
    if not isinstance(inputs_dict["house_ownership"], str):
        invalid_strings.append("House Ownership")
    if not isinstance(inputs_dict["car_ownership"], str):
        invalid_strings.append("Car Ownership")
    if not isinstance(inputs_dict["profession"], str):
        invalid_strings.append("Profession")
    if not isinstance(inputs_dict["city"], str):
        invalid_strings.append("City")
    if not isinstance(inputs_dict["state"], str):
        invalid_strings.append("State")
    if len(invalid_strings) == 1:
        invalid_datatype_message += f"{invalid_strings[0]} must be a string."
    if len(invalid_strings) > 1:
        invalid_datatype_message += f"{', '.join(invalid_strings[:-1])} and {invalid_strings[-1]} must be strings."

    if invalid_numbers or invalid_strings:
        return invalid_datatype_message
    return None  # no invalid data types


# Check out-of-range values in the inputs dictionary
def check_out_of_range_values(inputs_dict):
    out_of_range_inputs = []
    if inputs_dict["age"] < 21 or inputs_dict["age"] > 79:
        out_of_range_inputs.append("Age must be 21-79")
    if inputs_dict["married"] not in MARRIED_LABELS:
        out_of_range_inputs.append("Married/Single must be 'Single' or 'Married'")
    if inputs_dict["income"] < 0:
        out_of_range_inputs.append("Income must be a non-negative number")
    if inputs_dict["car_ownership"] not in CAR_OWNERSHIP_LABELS:
        out_of_range_inputs.append("Car Ownership must be 'Yes' or 'No'")
    if inputs_dict["house_ownership"] not in HOUSE_OWNERSHIP_LABELS:
        out_of_range_inputs.append("House Ownership must be 'Rented', 'Owned', or 'Neither Rented Nor Owned'")
    if inputs_dict["current_house_yrs"] < 10 or inputs_dict["current_house_yrs"] > 14:
        out_of_range_inputs.append("Current House Years must be 10-14")
    if inputs_dict["city"] not in CITY_LABELS:
        out_of_range_inputs.append("City must be one of the predefined cities")
    if inputs_dict["state"] not in STATE_LABELS:
        out_of_range_inputs.append("State must be one of the predefined states")
    if inputs_dict["profession"] not in PROFESSION_LABELS:
        out_of_range_inputs.append("Profession must be one of the predefined professions")
    if inputs_dict["experience"] < 0 or inputs_dict["experience"] > 20:
        out_of_range_inputs.append("Experience must be 0-20 years")
    if inputs_dict["current_job_yrs"] < 0 or inputs_dict["current_job_yrs"] > 14:
        out_of_range_inputs.append("Current Job Years must be 0-14")
    if len(out_of_range_inputs) == 1:
        return f"Out-of-range value error: {out_of_range_inputs[0]}."
    if len(out_of_range_inputs) > 1:
        return f"Out-of-range value error: {', '.join(out_of_range_inputs[:-1])} and {out_of_range_inputs[-1]}."
    return None  # no out-of-range inputs


# --- Load Machine Learning Pipeline ---
# Get the path to the pipeline file relative to this script
base_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_path = os.path.join(base_dir, "..", "models", "loan_default_rf_pipeline.pkl")

# Load the pre-trained pipeline (including data preprocessing and Random Forest Classifier model)
with open(pipeline_path, "rb") as file:
    pipeline = pickle.load(file)


# --- Function: Predict Probabilities of Loan Default with Pipeline ---
def _pipeline_predict_proba(input_df):
    return pipeline.predict_proba(input_df)


# --- Function: Predict Loan Default for a Single User ---
def single_predict(age, married, income, car_ownership, house_ownership, current_house_yrs, city, state, profession, experience, current_job_yrs):
    try:
        # --- Input preprocessing (part 1) ---
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

        # Format string inputs in snake_case
        inputs = snake_case_format_inputs(inputs)

        # Format "house_ownership" label as expected by pipeline 
        inputs["house_ownership"] = format_house_ownership(inputs["house_ownership"])

        # --- Input validation ---
        # Missing value check
        missing_value_message = check_missing_values(inputs)
        if missing_value_message:
            return missing_value_message, ""
        
        # Data type validation
        invalid_datatype_message = validate_data_types(inputs)
        if invalid_datatype_message:
            return invalid_datatype_message, ""

        # Out-of-range value check
        out_of_range_value_message = check_out_of_range_values(inputs)
        if out_of_range_value_message:
            return out_of_range_value_message, ""

        # --- Input preprocessing (part 2) ---
        # Convert float inputs to int 
        inputs["income"] = convert_float_to_int(inputs["income"])
        inputs["age"] = convert_float_to_int(inputs["age"])
        inputs["experience"] = convert_float_to_int(inputs["experience"])
        inputs["current_job_yrs"] = convert_float_to_int(inputs["current_job_yrs"])
        inputs["current_house_yrs"] = convert_float_to_int(inputs["current_house_yrs"])

        # Create input DataFrame for pipeline
        pipeline_input_df = pd.DataFrame([inputs])   
        
        # --- Predict loan default ---       
        # Use pipeline to predict probabilities 
        pred_proba = _pipeline_predict_proba(pipeline_input_df)

        # Create predicted probabilities dictionary (for gr.Label output)
        pred_proba_dict = {
            "Default": pred_proba[0, 1],  # "Default" is class 1
            "No Default": pred_proba[0, 0]  # "No Default" is class 0
        }

        # Apply optimized threshold to convert probabilities to binary predictions
        optimized_threshold = 0.29  # see threshold optimization in training script "loan_default_prediction.ipynb"
        prediction_int = (pred_proba[0, 1] >= optimized_threshold).astype(int)
        prediction_str_map = {0: "No Default", 1: "Default"}
        prediction_str = f"{prediction_str_map[prediction_int]}"

        return prediction_str, pred_proba_dict

    except Exception as e:
        return f"Error: {str(e)}", ""


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
        single_predict,
        inputs=[
            age, married, income, car_ownership, house_ownership, current_house_yrs, 
            city, state, profession, experience, current_job_yrs
        ],
        outputs=[prediction_text, pred_proba]
    )


# Launch the app
if __name__ == "__main__":
    app.launch()