import pytest
from selenium import webdriver


# End-to-end test that simulates a user filling out the form and receiving a prediction in the frontend UI 
@pytest.mark.e2e
def test_user_submits_loan_prediction_form():
    # Create a Chrome webdriver
    driver = webdriver.Chrome()

    try:
        # Get request to frontend Gradio UI running locally (make sure to run the app first)
        driver.get("http://localhost:7860")

    finally:
        # Close Chrome browser window
        driver.quit()
