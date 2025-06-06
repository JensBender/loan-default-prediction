# Imports
import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Add the parent directory to the path
from app.app import check_missing_values, predict_loan_default


class TestApp(unittest.TestCase):
    def test_check_missing_values(self):
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

# --- Manual tests ---
# 1 invalid numerical type
# print(predict_loan_default(50000, "five", 10, "Married", "Owned", "Yes", "Engineer", "Delhi", "Delhi", 5, 12)[0])
# 2 invalid numerical types
# print(predict_loan_default("50K", "five", 10, "Married", "Owned", "Yes", "Engineer", "Delhi", "Delhi", 5, 12)[0])
# 1 invalid string type
# print(predict_loan_default(50000, 30, 10, "Single", "Owned", 1, "Engineer", "Delhi", "Delhi", 5, 12)[0])
# 2 invalid string types
# print(predict_loan_default(50000, 30, 10, "Single", "Owned", False, 12345, "Delhi", "Delhi", 5, 12)[0])
# 3 invalid string types
# print(predict_loan_default(50000, 30, 10, "Single", "Owned", "Yes", 12345, 12345, 12345, 5, 12)[0])
