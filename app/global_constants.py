# --- Global constants for data preprocessing and model pipeline ---
# Imports
import numpy as np

# Define critical vs. non-critical features (for custom MissingValueChecker)
CRITICAL_FEATURES = ["income", "age", "experience", "profession", "city", "state", "current_job_yrs", "current_house_yrs"]
NON_CRITICAL_FEATURES = ["married", "car_ownership", "house_ownership"]

# Define columns to format categorical labels as snake case (for custom SnakeCaseFormatter)
COLUMNS_FOR_SNAKE_CASING = ["profession", "city", "state"]

# Map binary categorical columns to boolean (for custom BooleanColumnTransformer)
BOOLEAN_COLUMN_MAPPINGS = {
    "married": {"married": True, "single": False},
    "car_ownership": {"yes": True, "no": False}
}

# Map profession to job stability tier (for custom JobStabilityTransformer)
JOB_STABILITY_MAP = {
    # Government and highly regulated roles with exceptional job security
    "civil_servant": "very_stable",
    "army_officer": "very_stable",
    "police_officer": "very_stable",
    "magistrate": "very_stable",
    "official": "very_stable",
    "air_traffic_controller": "very_stable",
    "firefighter": "very_stable",
    "librarian": "very_stable",
    
    # Licensed/regulated professionals with strong job security
    "physician": "stable",
    "surgeon": "stable",
    "dentist": "stable",
    "chartered_accountant": "stable",
    "civil_engineer": "stable",
    "mechanical_engineer": "stable",
    "chemical_engineer": "stable",
    "petroleum_engineer": "stable",
    "biomedical_engineer": "stable",
    "engineer": "stable",
    
    # Corporate roles with steady demand
    "software_developer": "moderate",
    "computer_hardware_engineer": "moderate",
    "financial_analyst": "moderate",
    "industrial_engineer": "moderate",
    "statistician": "moderate",
    "microbiologist": "moderate",
    "scientist": "moderate",
    "geologist": "moderate",
    "economist": "moderate",
    "technology_specialist": "moderate",
    "design_engineer": "moderate",
    "architect": "moderate",
    "surveyor": "moderate",
    "secretary": "moderate",
    "flight_attendant": "moderate",
    "hotel_manager": "moderate",
    "computer_operator": "moderate",
    "technician": "moderate",
    
    # Project-based or variable demand roles
    "web_designer": "variable",
    "fashion_designer": "variable",
    "graphic_designer": "variable",
    "designer": "variable",
    "consultant": "variable",
    "technical_writer": "variable",
    "artist": "variable",
    "comedian": "variable",
    "chef": "variable",
    "analyst": "variable",
    "psychologist": "variable",
    "drafter": "variable",
    "aviator": "variable",
    "politician": "variable",
    "lawyer": "variable"
}

# Map city to city tier (for custom CityTierTransformer)
CITY_TIER_MAP = {
    # Tier 1 cities
    "new_delhi": "tier_1",
    "navi_mumbai": "tier_1",
    "kolkata": "tier_1",
    "bangalore": "tier_1",
    "chennai": "tier_1",
    "hyderabad": "tier_1",
    "mumbai": "tier_1",
    "pune": "tier_1",
    "ahmedabad": "tier_1",
    "jaipur": "tier_1",
    "lucknow": "tier_1",
    "noida": "tier_1",
    "coimbatore": "tier_1",
    "surat": "tier_1",
    "nagpur": "tier_1",
    "kochi": "tier_1",
    "thiruvananthapuram": "tier_1",
    "kanpur": "tier_1",
    "patna": "tier_1",
    
    # Tier 2 cities
    "bhopal": "tier_2",
    "vijayawada": "tier_2",
    "indore": "tier_2",
    "jodhpur": "tier_2",
    "vadodara": "tier_2",
    "ludhiana": "tier_2",
    "madurai": "tier_2",
    "agra": "tier_2",
    "mysore[7][8][9]": "tier_2",
    "rajkot": "tier_2",
    "nashik": "tier_2",
    "amritsar": "tier_2",
    "ranchi": "tier_2",
    "chandigarh_city": "tier_2",
    "allahabad": "tier_2",
    "bhubaneswar": "tier_2",
    "varanasi": "tier_2",
    "jabalpur": "tier_2",
    "guwahati": "tier_2",
    "tiruppur": "tier_2",
    "raipur": "tier_2",
    "udaipur": "tier_2",
    "gwalior": "tier_2",
    
    # Tier 3 cities
    "vijayanagaram": "tier_3",
    "bulandshahr": "tier_3",
    "saharsa[29]": "tier_3",
    "hajipur[31]": "tier_3",
    "satara": "tier_3",
    "ongole": "tier_3",
    "bellary": "tier_3",
    "giridih": "tier_3",
    "hospet": "tier_3",
    "khammam": "tier_3",
    "danapur": "tier_3",
    "bareilly": "tier_3",
    "satna": "tier_3",
    "howrah": "tier_3",
    "thanjavur": "tier_3",
    "farrukhabad": "tier_3",
    "buxar[37]": "tier_3",
    "arrah": "tier_3",
    "thrissur": "tier_3",
    "proddatur": "tier_3",
    "bahraich": "tier_3",
    "nandyal": "tier_3",
    "siwan[32]": "tier_3",
    "barasat": "tier_3",
    "dhule": "tier_3",
    "begusarai": "tier_3",
    "khandwa": "tier_3",
    "guntakal": "tier_3",
    "latur": "tier_3",
    "karaikudi": "tier_3"
}

# Define semantic column types (for ColumnTransformer to scale numerical and encode categorical columns)
NUMERICAL_COLUMNS = ["income", "age", "experience", "current_job_yrs", "current_house_yrs", "state_default_rate"]
CATEGORICAL_COLUMNS = ["house_ownership", "job_stability", "city_tier", "profession", "city", "state"]
BOOLEAN_COLUMNS = ["risk_flag", "married", "car_ownership"]

# Define the explicit order of categories for all ordinal columns (for OrdinalEncoder)
ORDINAL_COLUMN_ORDERS = [
    ["variable", "moderate", "stable", "very_stable"],  # Order for job_stability
    ["unknown", "tier_3", "tier_2", "tier_1"]  # Order for city_tier
]

# Define the columns to keep after preprocessing as model input (for custom FeatureSelector)
COLUMNS_TO_KEEP = [
    "income", "age", "experience", "current_job_yrs", "current_house_yrs", "state_default_rate", "house_ownership_owned", 
    "house_ownership_rented", "job_stability", "city_tier", "married", "car_ownership"
]

# Store the best Random Forest hyperparameter values identified with random search (for RandomForestClassifier)
RF_BEST_PARAMS = {
  "n_estimators": 225,
  "max_depth": 26,
  "min_samples_split": 2,
  "min_samples_leaf": 1,
  "max_features": np.float64(0.12974565961049356),
  "class_weight": "balanced"
}