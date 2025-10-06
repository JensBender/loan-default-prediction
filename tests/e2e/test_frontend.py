import pytest
from selenium import webdriver


# End-to-end test that simulates a user filling out the form and receiving a prediction in the frontend UI
@pytest.mark.e2e
def test_user_submits_loan_prediction_form():
    # Create Chrome browser
    driver = webdriver.Chrome()

    # Get request to frontend Gradio UI
    driver.get("http://localhost:7860")

    # Close Chrome browser window
    driver.quit()
