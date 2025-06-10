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
    assert check_missing_values(None, "Single", 60000, None, "Yes", 10, "Mumbai", "Maharashtra", "Manager", 5, 4) == "Please provide: Age and Car Ownership."
    assert check_missing_values(40, None, 80000, "Yes", None, 15, "Bangalore", "Karnataka", "Developer", 8, 6) == "Please provide: Married/Single and House Ownership."
    assert check_missing_values(30, "Single", None, "No", "Yes", None, "Mumbai", "Maharashtra", "Manager", 5, 4) == "Please provide: Income and Current House Years."
    assert check_missing_values(35, "Married", 70000, None, "No", 8, None, "Delhi", "Engineer", 4, 3) == "Please provide: Car Ownership and City."
    assert check_missing_values(28, "Single", 55000, "Yes", "No", 6, "Chennai", None, "Developer", None, 2) == "Please provide: State and Current House Years."
    assert check_missing_values(None, None, 50000, "Yes", "No", 10, "Mumbai", "Maharashtra", "Developer", 5, 3) == "Please provide: Age and Married/Single."
    assert check_missing_values(30, "Single", None, None, "No", 10, "Mumbai", "Maharashtra", "Developer", 5, 3) == "Please provide: Income and Car Ownership."
    assert check_missing_values(30, "Single", 60000, "Yes", None, None, "Mumbai", "Maharashtra", "Developer", 5, 3) == "Please provide: House Ownership and Current House Years."
    assert check_missing_values(30, "Single", 60000, "Yes", "No", 10, None, None, "Developer", 5, 3) == "Please provide: City and State."
    assert check_missing_values(30, "Single", 60000, "Yes", "No", 10, "Mumbai", "Maharashtra", None, None, 3) == "Please provide: Profession and Experience."
    assert check_missing_values(30, "Single", 60000, "Yes", "No", 10, "Mumbai", "Maharashtra", "Developer", None, None) == "Please provide: Experience and Current Job Years."   

    # All missing values
    assert check_missing_values(None, None, None, None, None, None, None, None, None, None, None) == "Please provide: Age, Married/Single, Income, Car Ownership, House Ownership, Current House Years, City, State, Profession, Experience and Current Job Years."
