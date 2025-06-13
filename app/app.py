# Imports
import gradio as gr
import pandas as pd
from app.custom_transformers import (
    MissingValueChecker,
    CategoricalLabelStandardizer,
    BooleanColumnTransformer,
    JobStabilityTransformer,
    CityTierTransformer,
    StateDefaultRateTargetEncoder,
    FeatureSelector
)
import pickle
import os


# --- Global constants ---
# List of professions, cities, and states (in same format as training data)
PROFESSIONS = [
    "Air_traffic_controller", "Analyst", "Architect", "Army_officer", "Artist",
    "Aviator", "Biomedical_Engineer", "Chartered_Accountant", "Chef", "Chemical_engineer",
    "Civil_engineer", "Civil_servant", "Comedian", "Computer_hardware_engineer", "Computer_operator",
    "Consultant", "Dentist", "Design_Engineer", "Designer", "Drafter",
    "Economist", "Engineer", "Fashion_Designer", "Financial_Analyst", "Firefighter",
    "Flight_attendant", "Geologist", "Graphic_Designer", "Hotel_Manager", "Industrial_Engineer",
    "Lawyer", "Librarian", "Magistrate", "Mechanical_engineer", "Microbiologist",
    "Official", "Petroleum_Engineer", "Physician", "Police_officer", "Politician",
    "Psychologist", "Scientist", "Secretary", "Software_Developer", "Statistician",
    "Surgeon", "Surveyor", "Technical_writer", "Technician", "Technology_specialist",
    "Web_designer"
]

CITIES = [
   "Adoni", "Agartala", "Agra", "Ahmedabad", "Ahmednagar", "Aizawl",
    "Ajmer", "Akola", "Alappuzha", "Aligarh", "Allahabad", "Alwar",
    "Ambala", "Ambarnath", "Ambattur", "Amravati", "Amritsar", "Amroha",
    "Anand", "Anantapur", "Anantapuram[24]", "Arrah", "Asansol",
    "Aurangabad", "Aurangabad[39]", "Avadi", "Bahraich", "Ballia",
    "Bally", "Bangalore", "Baranagar", "Barasat", "Bardhaman",
    "Bareilly", "Bathinda", "Begusarai", "Belgaum", "Bellary",
    "Berhampore", "Berhampur", "Bettiah[33]", "Bhadravati", "Bhagalpur",
    "Bhalswa_Jahangir_Pur", "Bharatpur", "Bhatpara", "Bhavnagar",
    "Bhilai", "Bhilwara", "Bhimavaram", "Bhind", "Bhiwandi", "Bhiwani",
    "Bhopal", "Bhubaneswar", "Bhusawal", "Bidar", "Bidhannagar",
    "Bihar_Sharif", "Bijapur", "Bikaner", "Bilaspur", "Bokaro",
    "Bongaigaon", "Bulandshahr", "Burhanpur", "Buxar[37]",
    "Chandigarh_city", "Chandrapur", "Chapra", "Chennai",
    "Chinsurah", "Chittoor[28]", "Coimbatore", "Cuttack", "Danapur",
    "Darbhanga", "Davanagere", "Dehradun", "Dehri[30]", "Delhi_city",
    "Deoghar", "Dewas", "Dhanbad", "Dharmavaram", "Dhule", "Dibrugarh",
    "Dindigul", "Durg", "Durgapur", "Eluru[25]", "Erode[17]", "Etawah",
    "Faridabad", "Farrukhabad", "Fatehpur", "Firozabad", "Gandhidham",
    "Gandhinagar", "Gangtok", "Gaya", "Ghaziabad", "Giridih",
    "Gopalpur", "Gorakhpur", "Gudivada", "Gulbarga", "Guna",
    "Guntakal", "Guntur[13]", "Gurgaon", "Guwahati", "Gwalior",
    "Hajipur[31]", "Haldia", "Hapur", "Haridwar", "Hazaribagh",
    "Hindupur", "Hospet", "Hosur", "Howrah", "Hubli-Dharwad", 
    "Hyderabad", "Ichalkaranji", "Imphal", "Indore", "Jabalpur",
    "Jaipur", "Jalandhar", "Jalgaon", "Jalna", "Jamalpur[36]",
    "Jammu[16]", "Jamnagar", "Jamshedpur", "Jaunpur", "Jehanabad[38]",
    "Jhansi", "Jodhpur", "Jorhat", "Junagadh", "Kadapa[23]",
    "Kakinada", "Kalyan-Dombivli", "Kamarhati", "Kanpur",
    "Karawal_Nagar", "Karaikudi", "Karimnagar", "Karnal", "Katihar",
    "Katni", "Kavali", "Khammam", "Khandwa", "Kharagpur",
    "Khora,_Ghaziabad", "Kirari_Suleman_Nagar", "Kishanganj[35]",
    "Kochi", "Kolhapur", "Kolkata", "Kollam", "Korba", "Kota[6]",
    "Kottayam", "Kozhikode", "Kulti", "Kumbakonam", "Kurnool[18]",
    "Latur", "Loni", "Lucknow", "Ludhiana", "Machilipatnam",
    "Madanapalle", "Madhyamgram", "Madurai", "Mahbubnagar",
    "Maheshtala", "Malda", "Malegaon", "Mango", "Mangalore", "Mathura",
    "Mau", "Medininagar", "Meerut", "Mehsana", "Mira-Bhayandar",
    "Mirzapur", "Miryalaguda", "Moradabad", "Morbi", "Morena",
    "Motihari[34]", "Mumbai", "Munger", "Muzaffarnagar", "Muzaffarpur",
    "Mysore[7][8][9]", "Nadiad", "Nagaon", "Nagercoil", "Nagpur",
    "Naihati", "Nanded", "Nandyal", "Nangloi_Jat", "Narasaraopet",
    "Nashik", "Navi_Mumbai", "Nellore[14][15]", "New_Delhi", "Nizamabad",
    "Noida", "North_Dumdum", "Ongole", "Orai", "Ozhukarai", "Pali",
    "Pallavaram", "Panchkula", "Panipat", "Panihati", "Panvel",
    "Parbhani", "Patiala", "Patna", "Phagwara", "Phusro",
    "Pimpri-Chinchwad", "Pondicherry", "Proddatur", "Pudukkottai",
    "Pune", "Purnia[26]", "Raebareli", "Raichur", "Raiganj", "Raipur",
    "Rajahmundry[19][20]", "Rajkot", "Rajpur_Sonarpur",
    "Ramagundam[27]", "Ramgarh", "Rampur", "Ranchi", "Ratlam",
    "Raurkela_Industrial_Township", "Rewa", "Rohtak", "Rourkela",
    "Sagar", "Saharanpur", "Saharsa[29]", "Salem", "Sambalpur",
    "Sambhal", "Sangli-Miraj_&_Kupwad", "Sasaram[30]", "Satara",
    "Satna", "Secunderabad", "Serampore", "Shahjahanpur", "Shimla",
    "Shimoga", "Shivpuri", "Sikar", "Silchar", "Siliguri",
    "Singrauli", "Sirsa", "Siwan[32]", "Solapur", "Sonipat",
    "South_Dumdum", "Sri_Ganganagar", "Srikakulam", "Srinagar",
    "Sultan_Pur_Majra", "Surat", "Surendranagar_Dudhrej", "Suryapet",
    "Tadipatri", "Tadepalligudem", "Tenali", "Tezpur", "Thane",
    "Thanjavur", "Thiruvananthapuram", "Thoothukudi", "Thrissur",
    "Tinsukia", "Tiruchirappalli[10]", "Tirunelveli",
    "Tirupati[21][22]", "Tiruppur", "Tiruvottiyur", "Tumkur",
    "Udaipur", "Udupi", "Ujjain", "Ulhasnagar", "Uluberia", "Unnao",
    "Vadodara", "Varanasi", "Vasai-Virar", "Vellore",
    "Vijayawada", "Vijayanagaram", "Visakhapatnam[4]",
    "Warangal[11][12]", "Yamunanagar",    
]

STATES = [
    "Andhra_Pradesh", "Assam", "Bihar", "Chandigarh", "Chhattisgarh",
    "Delhi", "Gujarat", "Haryana", "Himachal_Pradesh",
    "Jammu_and_Kashmir", "Jharkhand", "Karnataka", "Kerala",
    "Madhya_Pradesh", "Maharashtra", "Manipur", "Mizoram", "Odisha",
    "Puducherry", "Punjab", "Rajasthan", "Sikkim", "Tamil_Nadu",
    "Telangana", "Tripura", "Uttar_Pradesh", "Uttar_Pradesh[5]",
    "Uttarakhand", "West_Bengal",
]

# Format professions, cities, and states for display
professions = [profession.replace("_", " ").title() for profession in PROFESSIONS]
cities = [city.replace("_", " ").title() for city in CITIES]
states = [state.replace("_", " ").title() for state in STATES]


# Function to strip whitespace in inputs
def strip_whitespace(inputs_dict):
    return {
        key: value.strip() if isinstance(value, str) else value 
        for key, value in inputs_dict.items()
    }


# --- Input validation functions ---
def check_missing_values(inputs_dict):
    missing_inputs = []
    if inputs_dict["age"] in [None, "", [], {}, ()]:
        missing_inputs.append("Age")
    if not inputs_dict["married"]:  # catches None, "", 0, 0.0, False, [], {}, ()
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


def validate_data_types(age, married, income, car_ownership, house_ownership, current_house_yrs, city, state, profession, experience, current_job_yrs):
    invalid_numbers = []
    invalid_strings = []
    invalid_datatype_message = "Data type error! "   

    # Validate numerical inputs     
    if not isinstance(age, (int, float)):
        invalid_numbers.append("Age")
    if not isinstance(income, (int, float)):
        invalid_numbers.append("Income")
    if not isinstance(current_house_yrs, (int, float)):
        invalid_numbers.append("Current House Years")
    if not isinstance(experience, (int, float)):
        invalid_numbers.append("Experience")
    if not isinstance(current_job_yrs, (int, float)):
        invalid_numbers.append("Current Job Years")
    if len(invalid_numbers) == 1:
        invalid_datatype_message += f"{invalid_numbers[0]} must be a number."
    if len(invalid_numbers) > 1:
        invalid_datatype_message += f"{', '.join(invalid_numbers[:-1])} and {invalid_numbers[-1]} must be numbers."
    
    # Validate string inputs
    if not isinstance(married, str):
        invalid_strings.append("Married/Single")
    if not isinstance(house_ownership, str):
        invalid_strings.append("House Ownership")
    if not isinstance(car_ownership, str):
        invalid_strings.append("Car Ownership")
    if not isinstance(profession, str):
        invalid_strings.append("Profession")
    if not isinstance(city, str):
        invalid_strings.append("City")
    if not isinstance(state, str):
        invalid_strings.append("State")
    if len(invalid_strings) == 1:
        invalid_datatype_message += f"{invalid_strings[0]} must be a string."
    if len(invalid_strings) > 1:
        invalid_datatype_message += f"{', '.join(invalid_strings[:-1])} and {invalid_strings[-1]} must be strings."

    if invalid_numbers or invalid_strings:
        return invalid_datatype_message
    return None  # no invalid data types


# --- Load the pre-trained pipeline (including data preprocessing and Random Forest Classifier model) ---
# Get the path to the pipeline file relative to this script
base_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_path = os.path.join(base_dir, "..", "models", "loan_default_rf_pipeline.pkl")

# Load the pipeline
with open(pipeline_path, "rb") as file:
    pipeline = pickle.load(file)

# --- Function to predict loan default ---
def predict_loan_default(age, married, income, car_ownership, house_ownership, current_house_yrs, city, state, profession, experience, current_job_yrs):
    try:
        # Create inputs dictionary 
        inputs = {
            "age": age,
            "married": married,
            "income": income, 
            "car_ownership": car_ownership,
            "house_ownership": house_ownership,
            "current_house_yrs": current_house_yrs,
            "city": city,
            "state": state,
            "profession": profession,
            "experience": experience,
            "current_job_yrs": current_job_yrs
        }
        
        # Strip whitespace in inputs
        inputs = strip_whitespace(inputs)

        # --- Input validation ---
        # Missing value check
        missing_value_message = check_missing_values(inputs)
        if missing_value_message:
            return missing_value_message, ""
        
        # Data type validation
        invalid_datatype_message = validate_data_types(age, income, current_house_yrs, experience, current_job_yrs, married, house_ownership, car_ownership, profession, city, state)
        if invalid_datatype_message:
            return invalid_datatype_message, ""

        # Numerical input validation (must be non-negative integers or floats)
        invalid_numerical_inputs = []
        for numerical_input, input_value in {"Age": age, "Income": income, "Current House Years": current_house_yrs, "Experience": experience, "Current Job Years": current_job_yrs}.items():
            if not isinstance(input_value, (int, float)) or input_value < 0:
                invalid_numerical_inputs.append(numerical_input) 
        if len(invalid_numerical_inputs) == 1:
            return f"Error! {invalid_numerical_inputs[0]} must be a non-negative number.", ""
        if len(invalid_numerical_inputs) > 1:
            return f"Error! {', '.join(invalid_numerical_inputs[:-1])} and {invalid_numerical_inputs[-1]} must be non-negative numbers.", ""
        
        # Age validation (must be within training data range 21 to 79)
        if age < 21 or age > 79:
            return f"Note: The system doesn't currently support age {age}, as it is designed for applicants aged 21–79.", ""

        # --- Data preprocessing before pipeline ---
        # Convert numerical inputs from float (by default) to int to match expected pipeline input 
        income = int(round(income))
        age = int(round(age))
        experience = int(round(experience))
        current_job_yrs = int(round(current_job_yrs))
        current_house_yrs = int(round(current_house_yrs))

        # Convert UI categorical labels to match expected pipeline input
        married = married.lower()
        house_ownership = house_ownership.lower().replace("neither rented nor owned", "norent_noown")
        car_ownership = car_ownership.lower()

        # Create input DataFrame for Pipeline
        pipeline_input_df = pd.DataFrame({
            "income": [income],
            "age": [age],
            "experience": [experience],
            "married": [married],
            "house_ownership": [house_ownership],
            "car_ownership": [car_ownership],
            "profession": [profession],
            "city": [city],
            "state": [state],
            "current_job_yrs": [current_job_yrs],
            "current_house_yrs": [current_house_yrs]
        })   

        # Use single-row DataFrame as input
        pipeline_input_df = pipeline_input_df.head(1) 
        
        # --- Pipeline prediction ---       
        # Use pipeline to predict probabilities 
        pred_proba = pipeline.predict_proba(pipeline_input_df)

        # Use only first row of predictions 
        pred_proba = pred_proba[0, :] 

        # Create predicted probabilities dictionary (for gr.Label output)
        pred_proba_dict = {
            "Default": pred_proba[1],  # "Default" is class 1
            "No Default": pred_proba[0]  # "No Default" is class 0
        }

        # Apply optimized threshold to convert probabilities to binary predictions
        optimized_threshold = 0.29  # see threshold optimization in training script: loan_default_prediction.ipynb
        pred = (pred_proba[1] >= optimized_threshold).astype(int)

        # Create prediction text
        prediction_label_map = {0: "No Default", 1: "Default"}
        prediction_text = f"{prediction_label_map[pred]}"

        return prediction_text, pred_proba_dict
    
    except Exception as e:
        return f"Error: {str(e)}", ""


# --- Gradio app UI ---
# Custom CSS 
custom_css = """
.narrow-centered-column {
    max-width: 700px; 
    width: 100%; 
    margin-left: auto; 
    margin-right: auto;
}
#prediction-text textarea {font-size: 1.8em; font-weight: bold; text-align: center;}
#pred-proba-label {margin-top: -15px;}
#markdown-note {margin-top: -13px;}
"""

# Create Gradio app UI using Blocks
with gr.Blocks(css=custom_css) as app:
    # App title and description
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
            married = gr.Dropdown(label="Married/Single", choices=["Single", "Married"], value=None)
            income = gr.Number(label="Income", value="")
        with gr.Row():
            car_ownership = gr.Dropdown(label="Car Ownership", choices=["Yes", "No"], value=None)
            house_ownership = gr.Dropdown(label="House Ownership", choices=["Rented", "Owned", "Neither Rented Nor Owned"], value=None)
            current_house_yrs = gr.Slider(label="Current House Years", minimum=10, maximum=14, step=1)
        with gr.Row():
            city = gr.Dropdown(label="City", choices=cities, value=None)
            state = gr.Dropdown(label="State", choices=states, value=None)
            profession = gr.Dropdown(label="Profession", choices=professions, value=None)
        with gr.Row():
            experience = gr.Slider(label="Experience", minimum=0, maximum=20, step=1)
            current_job_yrs = gr.Slider(label="Current Job Years", minimum=0, maximum=14, step=1)
            gr.Markdown("")  # empty space for layout

    # Predict button and outputs
    with gr.Column(elem_classes="narrow-centered-column"):
        predict = gr.Button("Predict")
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

# Launch the app
if __name__ == "__main__":
    app.launch()