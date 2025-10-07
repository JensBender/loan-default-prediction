# Standard library imports
import time 

# Third-party library imports
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
        # Enter married
        married_field = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[aria-label='Married/Single']")))
        # Enter income
        income_field = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[aria-label='Income']")))
        income_field.send_keys(300000)
        # Enter car_ownership
        car_ownership_field = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[aria-label='Car Ownership']")))
        # Enter house_ownership
        house_ownership_field = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[aria-label='House Ownership']")))
        # Enter current_house_yrs
        current_house_yrs_field = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[aria-label='Current House Years']")))
        current_house_yrs_field.send_keys(11)
        # Enter city
        city_field = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[aria-label='City']")))
        # Enter state
        state_field = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[aria-label='State']")))
        # Enter profession
        profession_field = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[aria-label='Profession']")))
        # Enter experience
        experience_field = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[aria-label='Experience']")))
        experience_field.send_keys(3)
        # Enter current_job_yrs
        current_job_yrs_field = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[aria-label='Current Job Years']")))
        current_job_yrs_field.send_keys(3)

    finally:
        time.sleep(3)  # remove after dev/test phase
        # Close Chrome browser window
        driver.quit()
