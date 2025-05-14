import gradio as gr

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

states = [
    "Andhra_Pradesh", "Assam", "Bihar", "Chandigarh", "Chhattisgarh",
    "Delhi", "Gujarat", "Haryana", "Himachal_Pradesh",
    "Jammu_and_Kashmir", "Jharkhand", "Karnataka", "Kerala",
    "Madhya_Pradesh", "Maharashtra", "Manipur", "Mizoram", "Odisha",
    "Puducherry", "Punjab", "Rajasthan", "Sikkim", "Tamil_Nadu",
    "Telangana", "Tripura", "Uttar_Pradesh", "Uttar_Pradesh[5]",
    "Uttarakhand", "West_Bengal",
]


def predict_loan_default(income, age, experience, profession, state):
    return f"""
    Income: {income}
    Age: {age}
    Experience: {experience}
    Profession: {profession}
    State: {state}
    
    To Do: Implement loan default prediction model...
    """


app = gr.Interface(
    fn=predict_loan_default,
    inputs=[
        gr.Number(label="Income"), 
        gr.Number(label="Age"),
        gr.Number(label="Experience"),
        gr.Dropdown(label="Select Profession...", choices=professions, value=None),
        gr.Dropdown(label="Select State...", choices=states, value=None),
        ],
    outputs=gr.Textbox(),
    title="Loan Default Prediction",
    description="An app to predict loan default based on customer application data powered by machine learning."
    )

app.launch()