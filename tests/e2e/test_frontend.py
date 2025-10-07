import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC 


# End-to-end test that simulates a user filling out the form and receiving a prediction in the frontend UI 
@pytest.mark.e2e
def test_user_submits_loan_prediction_form():
    # Create a Chrome webdriver
    driver = webdriver.Chrome()

    try:
        # Get request to frontend Gradio UI running locally (make sure to run the app first)
        driver.get("http://localhost:7860")

        # Enter age
        age_field = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[aria-label='Age']")))
        age_field.send_keys(30)

    finally:
        # Close Chrome browser window
        driver.quit()
