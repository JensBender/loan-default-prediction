# Imports
import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Add the parent directory to the path
from app.app import check_missing_values


def test_check_missing_values():   
    # No missing values
    assert check_missing_values(30, "Single", 60000, "Yes", "No", 10, "Mumbai", "Maharashtra", "Developer", 5, 3) == None

    # 1 missing value
    assert check_missing_values(None, "Married", 50000, "Yes", "No", 5, "Delhi", "Delhi", "Engineer", 3, 2) == "Please provide: Age."
    assert check_missing_values(25, None, 50000, "Yes", "No", 5, "Delhi", "Delhi", "Engineer", 3, 2) == "Please provide: Married/Single."
    assert check_missing_values(25, "Married", None, "Yes", "No", 5, "Delhi", "Delhi", "Engineer", 3, 2) == "Please provide: Income."
    assert check_missing_values(25, "Married", 50000, None, "No", 5, "Delhi", "Delhi", "Engineer", 3, 2) == "Please provide: Car Ownership."
    assert check_missing_values(25, "Married", 50000, "Yes", None, 5, "Delhi", "Delhi", "Engineer", 3, 2) == "Please provide: House Ownership."
    assert check_missing_values(25, "Married", 50000, "Yes", "No", None, "Delhi", "Delhi", "Engineer", 3, 2) == "Please provide: Current House Years."
    assert check_missing_values(25, "Married", 50000, "Yes", "No", 5, None, "Delhi", "Engineer", 3, 2) == "Please provide: City."
    assert check_missing_values(25, "Married", 50000, "Yes", "No", 5, "Delhi", None, "Engineer", 3, 2) == "Please provide: State."
    assert check_missing_values(25, "Married", 50000, "Yes", "No", 5, "Delhi", "Delhi", None, 3, 2) == "Please provide: Profession."
    assert check_missing_values(25, "Married", 50000, "Yes", "No", 5, "Delhi", "Delhi", "Engineer", None, 2) == "Please provide: Experience."
    assert check_missing_values(25, "Married", 50000, "Yes", "No", 5, "Delhi", "Delhi", "Engineer", 3, None) == "Please provide: Current Job Years."

    # 2 missing values
    assert check_missing_values(25, "Married", None, "Yes", "No", 5, "Delhi", "Delhi", "Engineer", 3, None) == "Please provide: Income and Current Job Years."

    # 3 missing values
    assert check_missing_values(30, "Single", 60000, "Yes", "No", 10, None, None, None, 5, 3) == "Please provide: City, State and Profession."

    # 4 missing values
    assert check_missing_values(30, "Single", 60000, "Yes", "No", 10, None, None, None, None, 3) == "Please provide: City, State, Profession and Experience."

    # 5 missing values
    assert check_missing_values(30, "Single", 60000, "Yes", "No", None, None, None, None, None, 3) == "Please provide: Current House Years, City, State, Profession and Experience."

    # 6 missing values
    assert check_missing_values(30, "Single", 60000, "Yes", "No", None, None, None, None, None, None) == "Please provide: Current House Years, City, State, Profession, Experience and Current Job Years."

    # 7 missing values
    assert check_missing_values(None, None, None, None, None, None, None, "Maharashtra", "Developer", 5, 3) == "Please provide: Age, Married/Single, Income, Car Ownership, House Ownership, Current House Years and City."

    # 8 missing values
    assert check_missing_values(None, None, None, None, None, None, "Mumbai", "Maharashtra", None, None, 3) == "Please provide: Age, Married/Single, Income, Car Ownership, House Ownership, Current House Years, Profession and Experience."
 
    # 9 missing values
    assert check_missing_values(None, None, None, None, None, None, None, None, "Developer", None, 3) == "Please provide: Age, Married/Single, Income, Car Ownership, House Ownership, Current House Years, City, State and Experience."
 
    # 10 missing values
    assert check_missing_values(None, None, None, None, None, None, None, None, "Developer", None, None) == "Please provide: Age, Married/Single, Income, Car Ownership, House Ownership, Current House Years, City, State, Experience and Current Job Years."
 
    # All missing values
    assert check_missing_values(None, None, None, None, None, None, None, None, None, None, None) == "Please provide: Age, Married/Single, Income, Car Ownership, House Ownership, Current House Years, City, State, Profession, Experience and Current Job Years."
