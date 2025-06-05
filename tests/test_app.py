# Set the system path to the root directory (one up from the current directory)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  

# Import the predict_loan_default function from the app
from app.app import predict_loan_default

# --- Input validation tests ---
# 1 invalid numerical type
print(predict_loan_default(50000, "five", 10, "Married", "Owned", "Yes", "Engineer", "Delhi", "Delhi", 5, 12)[0])
# 2 invalid numerical types
print(predict_loan_default("50K", "five", 10, "Married", "Owned", "Yes", "Engineer", "Delhi", "Delhi", 5, 12)[0])
# 1 invalid string type
print(predict_loan_default(50000, 30, 10, "Single", "Owned", 1, "Engineer", "Delhi", "Delhi", 5, 12)[0])
# 2 invalid string types
print(predict_loan_default(50000, 30, 10, "Single", "Owned", False, 12345, "Delhi", "Delhi", 5, 12)[0])
# 3 invalid string types
print(predict_loan_default(50000, 30, 10, "Single", "Owned", "Yes", 12345, 12345, 12345, 5, 12)[0])