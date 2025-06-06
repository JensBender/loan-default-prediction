# Imports
import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Add the parent directory to the path
from app.app import check_missing_values


def test_check_no_missing_values():
    assert check_missing_values(25, "Married", 50000, "Yes", "No", 5, "Delhi", "Delhi", "Engineer", 3, 2) == None
    assert check_missing_values(30, "Single", 60000, "No", "Yes", 10, "Mumbai", "Maharashtra", "Manager", 5, 4) == None
    assert check_missing_values(40, "Married", 80000, "Yes", "No", 15, "Bangalore", "Karnataka", "Developer", 8, 6) == None


def test_check_single_missing_values():   
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


def test_check_multiple_missing_values():
    assert check_missing_values(None, None, None, None, None, None, None, None, None, None, None) == "Please provide: Age, Married/Single, Income, Car Ownership, House Ownership, Current House Years, City, State, Profession, Experience and Current Job Years."
    assert check_missing_values(25, "Married", None, "Yes", "No", 5, "Delhi", "Delhi", "Engineer", 3, None) == "Please provide: Income and Current Job Years."
    assert check_missing_values(None, "Single", 60000, None, "Yes", 10, "Mumbai", "Maharashtra", "Manager", 5, 4) == "Please provide: Age and Car Ownership."
    assert check_missing_values(40, None, 80000, "Yes", None, 15, "Bangalore", "Karnataka", "Developer", 8, 6) == "Please provide: Married/Single and House Ownership."
    assert check_missing_values(30, "Single", None, "No", "Yes", None, "Mumbai", "Maharashtra", "Manager", 5, 4) == "Please provide: Income and Current House Years."
