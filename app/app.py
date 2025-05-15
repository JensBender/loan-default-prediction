# Imports
import gradio as gr

# List of professions, cities, and states for dropdown 
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


# Function to predict loan default based on customer application data using Random Forest Classifier
def predict_loan_default(income, age, experience, profession, city, state):
    return f"""
    Income: {income}
    Age: {age}
    Experience: {experience}
    Profession: {profession}
    City: {city}
    State: {state}
    
    To Do: Implement loan default prediction model...
    """


# Gradio app interface
app = gr.Interface(
    fn=predict_loan_default,
    inputs=[
        gr.Number(label="Income"), 
        gr.Number(label="Age"),
        gr.Number(label="Experience"),
        gr.Dropdown(label="Select Profession...", choices=professions, value=None),
        gr.Dropdown(label="Select City...", choices=cities, value=None),
        gr.Dropdown(label="Select State...", choices=states, value=None),
        ],
    outputs=gr.Textbox(),
    title="Loan Default Prediction",
    description="An app to predict loan default based on customer application data powered by machine learning."
    )

# Launch the app
if __name__ == "__main__":
    app.launch()