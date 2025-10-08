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
def test_user_submits_loan_default_prediction_form():
    # Create a Chrome webdriver
    driver = webdriver.Chrome()

    try:
        # Get request to frontend Gradio UI running locally (make sure to run the app first)
        driver.get("http://localhost:7860")

        # --- Gradio Number inputs ---
        # Enter age
        age_field = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[aria-label='Age']")))
        age_field.send_keys(30)
        # Enter income
        income_field = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[aria-label='Income']")))
        income_field.send_keys(300000)

        # --- Gradio Slider inputs ---
        # Sliders have both number input and range slider, use number input (identified via aria-label)
        # Enter current_house_yrs
        current_house_yrs_field = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[aria-label='number input for Current House Years']")))
        current_house_yrs_field.clear()
        current_house_yrs_field.send_keys(11)
        # Enter experience
        experience_field = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[aria-label='number input for Experience']")))
        experience_field.clear()
        experience_field.send_keys(3)
        # Enter current_job_yrs
        current_job_yrs_field = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[aria-label='number input for Current Job Years']")))
        current_job_yrs_field.clear()
        current_job_yrs_field.send_keys(3)

        # --- Gradio Dropdown inputs ---
        # First click Dropdown to bring up the options, then click on an option 
        # Enter married
        married_dropdown = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[aria-label='Married/Single']")))
        married_dropdown.click()
        married_dropdown_option = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//ul[contains(@class, 'options')]//li[text()='Single']")))
        assert married_dropdown_option.text == "Single"
        married_dropdown_option.click()
        # Enter car_ownership
        car_ownership_field = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[aria-label='Car Ownership']")))
        car_ownership_field.click()
        car_ownership_field_option = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//ul[contains(@class, 'options')]//li[text()='No']")))
        assert car_ownership_field_option.text == "No"
        car_ownership_field_option.click()
        # Enter house_ownership
        house_ownership_field = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[aria-label='House Ownership']")))
        house_ownership_field.click()
        house_ownership_field_option = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//ul[contains(@class, 'options')]//li[text()='Neither Rented Nor Owned']")))
        assert house_ownership_field_option.text == "Neither Rented Nor Owned"
        house_ownership_field_option.click()
        # Enter city
        city_field = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[aria-label='City']")))
        city_field.click()
        city_field_option = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//ul[contains(@class, 'options')]//li[text()='Sikar']")))
        assert city_field_option.text == "Sikar"
        city_field_option.click()
        # Enter state
        state_field = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[aria-label='State']")))
        state_field.click()
        state_field_option = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//ul[contains(@class, 'options')]//li[text()='Rajasthan']")))
        assert state_field_option.text == "Rajasthan"
        state_field_option.click()
        # Enter profession
        profession_field = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[aria-label='Profession']")))
        profession_field.click()
        profession_field_option = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//ul[contains(@class, 'options')]//li[text()='Artist']")))
        assert profession_field_option.text == "Artist"
        profession_field_option.click()

        # Click predict button
        predict_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//div[@id='predict-button-wrapper']/button")))
        predict_button.click()

        # Prediction result
        # Find probability elements
        default_probability = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, "//dl[contains(@class, 'label')]//dt[text()='Default']/following-sibling::dd")))
        no_default_probability = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, "//dl[contains(@class, 'label')]//dt[text()='No Default']/following-sibling::dd")))
        # Extract numbers
        default_probability = int(default_probability.text.replace("%", ""))
        no_default_probability = int(no_default_probability.text.replace("%", ""))
        # Find prediction text element
        prediction = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//textarea[@placeholder='Prediction Result']")))

        print(default_probability)
        print(no_default_probability)
        print(prediction.text)

        # Ensure prediction is as expected
        assert prediction.text in ["Default", "No Default"]
        # Ensure probabilities are numbers between 0 and 100
        assert 0 <= default_probability <=100
        assert 0 <= no_default_probability <= 100
        # Ensure probabilities sum to approximately 100
        sum = default_probability + no_default_probability
        assert 99 <= sum <= 101  # allow for rounding edge cases
         
    finally:
        time.sleep(10)  # remove after dev/test phase
        # Close Chrome browser window
        driver.quit()
