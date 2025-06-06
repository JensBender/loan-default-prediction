# Imports
import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Add the parent directory to the path
from app.app import check_missing_values


class TestApp(unittest.TestCase):
    def test_check_missing_values(self):
        # Print error messages
        print(check_missing_values(25, "Married", 50000, "Yes", "No", 5, "Delhi", "Delhi", "Engineer", 3, 2))
        print(check_missing_values(None, "Married", 50000, "Yes", "No", 5, "Delhi", "Delhi", "Engineer", 3, 2))
        print(check_missing_values(25, None, 50000, "Yes", "No", 5, "Delhi", "Delhi", "Engineer", 3, 2))
        print(check_missing_values(25, "Married", None, "Yes", "No", 5, "Delhi", "Delhi", "Engineer", 3, 2))
        print(check_missing_values(25, "Married", 50000, None, "No", 5, "Delhi", "Delhi", "Engineer", 3, 2))
        print(check_missing_values(25, "Married", 50000, "Yes", None, 5, "Delhi", "Delhi", "Engineer", 3, 2))
        print(check_missing_values(25, "Married", 50000, "Yes", "No", None, "Delhi", "Delhi", "Engineer", 3, 2))
        print(check_missing_values(25, "Married", 50000, "Yes", "No", 5, None, "Delhi", "Engineer", 3, 2))
        print(check_missing_values(25, "Married", 50000, "Yes", "No", 5, "Delhi", None, "Engineer", 3, 2))
        print(check_missing_values(25, "Married", 50000, "Yes", "No", 5, "Delhi", "Delhi", None, 3, 2))
        print(check_missing_values(25, "Married", 50000, "Yes", "No", 5, "Delhi", "Delhi", "Engineer", None, 2))
        print(check_missing_values(25, "Married", 50000, "Yes", "No", 5, "Delhi", "Delhi", "Engineer", 3, None))
        
        # Assert error messages
        self.assertEqual(check_missing_values(25, "Married", 50000, "Yes", "No", 5, "Delhi", "Delhi", "Engineer", 3, 2), None)
        self.assertEqual(check_missing_values(None, "Married", 50000, "Yes", "No", 5, "Delhi", "Delhi", "Engineer", 3, 2), "Please provide: Age.")
        self.assertEqual(check_missing_values(25, None, 50000, "Yes", "No", 5, "Delhi", "Delhi", "Engineer", 3, 2), "Please provide: Married/Single.")
        self.assertEqual(check_missing_values(25, "Married", None, "Yes", "No", 5, "Delhi", "Delhi", "Engineer", 3, 2), "Please provide: Income.")
        self.assertEqual(check_missing_values(25, "Married", 50000, None, "No", 5, "Delhi", "Delhi", "Engineer", 3, 2), "Please provide: Car Ownership.")
        self.assertEqual(check_missing_values(25, "Married", 50000, "Yes", None, 5, "Delhi", "Delhi", "Engineer", 3, 2), "Please provide: House Ownership.")
        self.assertEqual(check_missing_values(25, "Married", 50000, "Yes", "No", None, "Delhi", "Delhi", "Engineer", 3, 2), "Please provide: Current House Years.")
        self.assertEqual(check_missing_values(25, "Married", 50000, "Yes", "No", 5, None, "Delhi", "Engineer", 3, 2), "Please provide: City.")
        self.assertEqual(check_missing_values(25, "Married", 50000, "Yes", "No", 5, "Delhi", None, "Engineer", 3, 2), "Please provide: State.")
        self.assertEqual(check_missing_values(25, "Married", 50000, "Yes", "No", 5, "Delhi", "Delhi", None, 3, 2), "Please provide: Profession.")
        self.assertEqual(check_missing_values(25, "Married", 50000, "Yes", "No", 5, "Delhi", "Delhi", "Engineer", None, 2), "Please provide: Experience.")
        self.assertEqual(check_missing_values(25, "Married", 50000, "Yes", "No", 5, "Delhi", "Delhi", "Engineer", 3, None), "Please provide: Current Job Years.")


if __name__ == "__main__":
    unittest.main()