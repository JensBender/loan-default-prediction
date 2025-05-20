# Imports
import gradio as gr
import pandas as pd

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
                return f"Error! {numerical_input} must be a non-negative number.", "Error", None
            
        # Age validation (must be within training data range 21 to 79)
        if age < 21 or age > 79:
            return f"Note: The automated loan default prediction system doesn't currently support age {age}, as it is designed for applicants aged 21–79.", "Error", None

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
            return f"Error! Please select: {', '.join(missing_inputs)}.", "Error", None
        
        # --- Data preprocessing ---
        # Format categorical label inputs to match expected model inputs 
        def standardize_categorical_labels(categorical_label):
            return (
                categorical_label
                .strip()  # Remove leading/trailing spaces
                .lower()  # Convert to lowercase
                .replace("-", "_")  # Replace hyphens with "_"
                .replace("/", "_")  # Replace slashes with "_"
                .replace(" ", "_")  # Replace spaces with "_"
            )   

        married = standardize_categorical_labels(married)
        house_ownership = standardize_categorical_labels(house_ownership.replace("Neither Rented Nor Owned", "norent_noown"))
        car_ownership = standardize_categorical_labels(car_ownership)
        profession = standardize_categorical_labels(profession)
        city = standardize_categorical_labels(city)
        state = standardize_categorical_labels(state)

        # Create DataFrame from inputs
        input_df = pd.DataFrame({
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
        prediction = {"Default": 0.3, "No Default": 0.7}  # Placeholder prediction for now

        return prediction, prediction, input_df
    
    except Exception as e:
        return f"Error: {str(e)}", "Error", None


# Gradio app interface
with gr.Blocks() as app:
    gr.Markdown(
        """
        <h1 style='text-align:center'>Loan Default Prediction</h1>
        <p style='text-align:center'>An automated loan default prediction system. Submit the customer application data to receive an automated prediction powered by machine learning.</p>
        """
    )
    with gr.Row():
        # Two input columns
        with gr.Column(scale=2):
            gr.Markdown("<h2>Input</h2>")
            with gr.Row():
                # Input column 1
                with gr.Column():
                    income = gr.Number(label="Income")
                    age = gr.Number(label="Age")
                    married = gr.Dropdown(label="Married/Single", choices=["Single", "Married"])
                    house_ownership = gr.Dropdown(label="House Ownership", choices=["Rented", "Owned", "Neither Rented Nor Owned"])
                    experience = gr.Slider(label="Experience", minimum=0, maximum=20, step=1)
                    current_job_yrs = gr.Slider(label="Current Job Years", minimum=0, maximum=14, step=1)
            # Input column 2
                with gr.Column():
                    car_ownership = gr.Dropdown(label="Car Ownership", choices=["Yes", "No"])
                    profession = gr.Dropdown(label="Profession", choices=professions)
                    city = gr.Dropdown(label="City", choices=cities)
                    state = gr.Dropdown(label="State", choices=states)
                    current_house_yrs = gr.Slider(label="Current House Years", minimum=10, maximum=14, step=1)
        # Output column
        with gr.Column(scale=1):
            gr.Markdown("<h2>Prediction</h2>")
            predict = gr.Button("Predict")
            prediction = gr.Label(label="Prediction")
            model_output = gr.Textbox(label="Model Output")
    with gr.Row():
        with gr.Column(scale=3):
            # Input Dataframe spanning across all three columns
            input_df = gr.Dataframe(label="Input Dataframe")

    predict.click(
        predict_loan_default,
        inputs=[
            income, age, experience, married, house_ownership, car_ownership,
            profession, city, state, current_job_yrs, current_house_yrs
        ],
        outputs=[prediction, model_output, input_df]
    )

# Launch the app
if __name__ == "__main__":
    app.launch()