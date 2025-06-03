# Imports
import gradio as gr
import pandas as pd
from custom_transformers import (
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

# List of professions, cities, and states (in same format as training data)
professions = [
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

cities = [
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

states = [
    "Andhra_Pradesh", "Assam", "Bihar", "Chandigarh", "Chhattisgarh",
    "Delhi", "Gujarat", "Haryana", "Himachal_Pradesh",
    "Jammu_and_Kashmir", "Jharkhand", "Karnataka", "Kerala",
    "Madhya_Pradesh", "Maharashtra", "Manipur", "Mizoram", "Odisha",
    "Puducherry", "Punjab", "Rajasthan", "Sikkim", "Tamil_Nadu",
    "Telangana", "Tripura", "Uttar_Pradesh", "Uttar_Pradesh[5]",
    "Uttarakhand", "West_Bengal",
]

# Format professions, cities, and states for display
professions = [profession.replace("_", " ").title() for profession in professions]
cities = [city.replace("_", " ").title() for city in cities]
states = [state.replace("_", " ").title() for state in states]


# Function to predict loan default based on customer application data using machine learning model
def predict_loan_default(income, age, experience, married, house_ownership, car_ownership, profession, city, state, current_job_yrs, current_house_yrs):
    try:
        # --- Input validation ---
        # Numerical input validation (must be non-negative integers or floats)
        for numerical_input, input_value in {"Income": income, "Age": age, "Experience": experience, "Current Job Years": current_job_yrs, "Current House Years": current_house_yrs}.items():
            if not isinstance(input_value, (int, float)) or input_value < 0:
                error_message = f"Error! {numerical_input} must be a non-negative number."
                return error_message, error_message, pd.DataFrame(), "Error"
            
        # Age validation (must be within training data range 21 to 79)
        if age < 21 or age > 79:
            error_message = f"Note: The automated loan default prediction system doesn't currently support age {age}, as it is designed for applicants aged 21–79."
            return error_message, error_message, pd.DataFrame(), "Error"

        # Missing input check
        missing_inputs = []
        if not married:
            missing_inputs.append("Married/Single")
        if not house_ownership:
            missing_inputs.append("House Ownership")
        if not car_ownership:
            missing_inputs.append("Car Ownership")
        if not profession:
            missing_inputs.append("Profession")
        if not city:
            missing_inputs.append("City")
        if not state:
            missing_inputs.append("State")
        if missing_inputs:
            return f"Error! Please select: {', '.join(missing_inputs)}.", f"Error! Please select: {', '.join(missing_inputs)}.", pd.DataFrame(), "Error"
        
        # --- Data preprocessing ---
        # Convert numerical inputs from float (by default) to int to match training data 
        income = int(round(income))
        age = int(round(age))
        experience = int(round(experience))
        current_job_yrs = int(round(current_job_yrs))
        current_house_yrs = int(round(current_house_yrs))

        # Convert UI categorical labels to match training data
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
        
        # --- Model prediction --- 
        # Get the path to the pipeline file relative to this script 
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pipeline_path = os.path.join(base_dir, "..", "models", "loan_default_rf_pipeline.pkl")

        # Load the pre-trained pipeline including data preprocessing and Random Forest Classifier model
        with open(pipeline_path, "rb") as file:
            pipeline = pickle.load(file)
        
        # Use single-row DataFrame as input
        pipeline_input_df = pipeline_input_df.head(1) 

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
        optimized_threshold = 0.29  # see training script: loan_default_prediction.ipynb
        pred = (pred_proba[1] >= optimized_threshold).astype(int)

        # Create prediction text
        prediction_label_map = {0: "No Default", 1: "Default"}
        prediction_text = f"{prediction_label_map[pred]}"

        return pred_proba_dict, prediction_text, pipeline_input_df, pred_proba
    
    except Exception as e:
        return f"Error: {str(e)}", f"Error: {str(e)}", pd.DataFrame(), "Error"


# --- Gradio app UI ---
# Custom CSS 
custom_css = """
.narrow-centered-column {
    max-width: 600px; 
    width: 100%; 
    margin-left: auto; 
    margin-right: auto;
}
#markdown-note {margin-top: -15px;}
"""

# Create Gradio app UI using Blocks
with gr.Blocks(css=custom_css) as app:
    # App title and description
    gr.Markdown(
        """
        <h1 style='text-align:center'>Loan Default Prediction</h1>
        <p style='text-align:center'>An automated loan default prediction system. Submit the customer application data to receive an automated prediction powered by machine learning.</p>
        """
    )
    
    # Input
    with gr.Group():
        with gr.Row():
            age = gr.Number(label="Age")
            married = gr.Dropdown(label="Married/Single", choices=["Single", "Married"])
            income = gr.Number(label="Income")
        with gr.Row():
            car_ownership = gr.Dropdown(label="Car Ownership", choices=["Yes", "No"])
            house_ownership = gr.Dropdown(label="House Ownership", choices=["Rented", "Owned", "Neither Rented Nor Owned"])
            current_house_yrs = gr.Slider(label="Current House Years", minimum=10, maximum=14, step=1)
        with gr.Row():
            city = gr.Dropdown(label="City", choices=cities)
            state = gr.Dropdown(label="State", choices=states)
            profession = gr.Dropdown(label="Profession", choices=professions)
        with gr.Row():
            experience = gr.Slider(label="Experience", minimum=0, maximum=20, step=1)
            current_job_yrs = gr.Slider(label="Current Job Years", minimum=0, maximum=14, step=1)
            gr.Markdown("")  # empty space for layout
   
    # Predict button and outputs
    with gr.Column(elem_classes="narrow-centered-column"):
        predict = gr.Button("Predict")
        prediction_text = gr.Textbox(show_label=False)
        pred_proba = gr.Label(show_label=False, show_heading=False)
        gr.Markdown(
            "<small>Note: Prediction uses an optimized decision threshold of 0.29 "
            "(predicts 'Default' if probability ≥ 29%, otherwise 'No Default').</small>",
            elem_id="markdown-note"
        )

    # Pipeline input and output for testing
    with gr.Row():
        pipeline_input_df = gr.Dataframe(label="Pipeline Input")
    with gr.Row():
        pipeline_output = gr.Textbox(label="Pipeline Output (pred_proba array)")

    # Predict button click event
    predict.click(
        predict_loan_default,
        inputs=[
            income, age, experience, married, house_ownership, car_ownership,
            profession, city, state, current_job_yrs, current_house_yrs
        ],
        outputs=[pred_proba, prediction_text, pipeline_input_df, pipeline_output]
    )


# Launch the app
if __name__ == "__main__":
    app.launch(debug=True)