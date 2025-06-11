# Imports
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress deprecation warnings
import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # add the parent directory to the path
from app.app import check_missing_values, validate_data_types


# Define valid input values for testing as dictionary
@pytest.fixture
def valid_inputs():
    return {
        "age": 30,
        "married": "Married",
        "income": 1000000,
        "car_ownership": "Yes",
        "house_ownership": "Rented",
        "current_house_yrs": 12,
        "city": "Delhi",
        "state": "Assam",
        "profession": "Architect",
        "experience": 10,
        "current_job_yrs": 7
    }


def test_check_no_missing_values(valid_inputs):
    assert check_missing_values(**valid_inputs) == None


def test_check_missing_values():   
    # 1 missing value
    assert check_missing_values(None, "Married", 1000000, "Yes", "Rented", 12, "Delhi", "Assam", "Architect", 10, 7) == "Please provide: Age."
    assert check_missing_values(30, None, 1000000, "Yes", "Rented", 12, "Delhi", "Assam", "Architect", 10, 7) == "Please provide: Married/Single."
    assert check_missing_values(30, "Married", None, "Yes", "Rented", 12, "Delhi", "Assam", "Architect", 10, 7) == "Please provide: Income."
    assert check_missing_values(30, "Married", 1000000, None, "Rented", 12, "Delhi", "Assam", "Architect", 10, 7) == "Please provide: Car Ownership."
    assert check_missing_values(30, "Married", 1000000, "Yes", None, 12, "Delhi", "Assam", "Architect", 10, 7) == "Please provide: House Ownership."
    assert check_missing_values(30, "Married", 1000000, "Yes", "Rented", None, "Delhi", "Assam", "Architect", 10, 7) == "Please provide: Current House Years."
    assert check_missing_values(30, "Married", 1000000, "Yes", "Rented", 12, None, "Assam", "Architect", 10, 7) == "Please provide: City."
    assert check_missing_values(30, "Married", 1000000, "Yes", "Rented", 12, "Delhi", None, "Architect", 10, 7) == "Please provide: State."
    assert check_missing_values(30, "Married", 1000000, "Yes", "Rented", 12, "Delhi", "Assam", None, 10, 7) == "Please provide: Profession."
    assert check_missing_values(30, "Married", 1000000, "Yes", "Rented", 12, "Delhi", "Assam", "Architect", None, 7) == "Please provide: Experience."
    assert check_missing_values(30, "Married", 1000000, "Yes", "Rented", 12, "Delhi", "Assam", "Architect", 10, None) == "Please provide: Current Job Years."

    # 2 missing values
    assert check_missing_values(30, "Married", None, "Yes", "Rented", 12, "Delhi", "Assam", "Architect", 10, None) == "Please provide: Income and Current Job Years."

    # 3 missing values
    assert check_missing_values(30, "Married", 1000000, "Yes", "Rented", 12, None, None, None, 10, 7) == "Please provide: City, State and Profession."

    # 4 missing values
    assert check_missing_values(30, "Married", 1000000, "Yes", "Rented", 12, None, None, None, None, 7) == "Please provide: City, State, Profession and Experience."

    # 5 missing values
    assert check_missing_values(30, "Married", 1000000, "Yes", "Rented", None, None, None, None, None, 7) == "Please provide: Current House Years, City, State, Profession and Experience."

    # 6 missing values
    assert check_missing_values(30, "Married", 1000000, "Yes", "Rented", None, None, None, None, None, None) == "Please provide: Current House Years, City, State, Profession, Experience and Current Job Years."

    # 7 missing values
    assert check_missing_values(None, None, None, None, None, None, None, "Assam", "Architect", 10, 7) == "Please provide: Age, Married/Single, Income, Car Ownership, House Ownership, Current House Years and City."

    # 8 missing values
    assert check_missing_values(None, None, None, None, None, None, "Delhi", "Assam", None, None, 7) == "Please provide: Age, Married/Single, Income, Car Ownership, House Ownership, Current House Years, Profession and Experience."
 
    # 9 missing values
    assert check_missing_values(None, None, None, None, None, None, None, None, "Architect", None, 7) == "Please provide: Age, Married/Single, Income, Car Ownership, House Ownership, Current House Years, City, State and Experience."
 
    # 10 missing values
    assert check_missing_values(None, None, None, None, None, None, None, None, "Architect", None, None) == "Please provide: Age, Married/Single, Income, Car Ownership, House Ownership, Current House Years, City, State, Experience and Current Job Years."
 
    # All missing values
    assert check_missing_values(None, None, None, None, None, None, None, None, None, None, None) == "Please provide: Age, Married/Single, Income, Car Ownership, House Ownership, Current House Years, City, State, Profession, Experience and Current Job Years."


# --- Inputs and their value ranges --- 
# age
# married ["Single", "Married"]
# income
# car_ownership ["Yes", "No"] 
# house_ownership ["Rented", "Owned", "Neither Rented Nor Owned"] 
# current_house_yrs [10-14]
# city ["Adoni", "Agartala", "Agra", "Ahmedabad", "Ahmednagar", ...]
# state ["Andhra_Pradesh", "Assam", "Bihar", "Chandigarh", "Chhattisgarh", "Delhi", ...]
# profession ["Air_traffic_controller", "Analyst", "Architect", "Army_officer", "Artist", ...]
# experience [0-20]
# current_job_yrs [0-14]


def test_validate_numerical_inputs():
    # No invalid data types
    assert validate_data_types(30, "Married", 1000000, "Yes", "Rented", 12, "Delhi", "Assam", "Architect", 10, 7) == None