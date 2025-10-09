# Standard library imports
import time 

# Third-party library imports
import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC 
from selenium.webdriver.remote.webdriver import WebDriver


# --- Helper Functions ---
# Make Gradio Number input
def make_number_input(webdriver: WebDriver, number_input: str, value: int) -> None:
    number_input_element = WebDriverWait(webdriver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, f"input[aria-label='{number_input}']")))
    number_input_element.send_keys(value)

# Make Gradio Slider input
def make_slider_input(webdriver: WebDriver, slider_input: str, value: int) -> None:
    # Sliders have both number input and range slider, use number input (identified via aria-label)
    slider_input_element = WebDriverWait(webdriver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, f"input[aria-label='number input for {slider_input}']")))
    slider_input_element.clear()
    slider_input_element.send_keys(value)

# Make Gradio Dropdown input
def make_dropdown_input(webdriver: WebDriver, dropdown_input: str, value: str) -> None:
    # First click Dropdown to bring up the options, then click on an option 
    dropdown_input_element = WebDriverWait(webdriver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, f"input[aria-label='{dropdown_input}']")))
    dropdown_input_element.click()
    dropdown_menu_option = WebDriverWait(webdriver, 10).until(EC.element_to_be_clickable((By.XPATH, f"//ul[contains(@class, 'options')]//li[text()='{value}']")))
    assert dropdown_menu_option.text == f"{value}"
    dropdown_menu_option.click()


# End-to-end happy path test that simulates a user submitting the form and receiving a prediction in the frontend UI 
@pytest.mark.e2e
def test_user_submits_loan_default_prediction_form():
    # Customize Chrome webdriver with options
    chrome_options = Options()
    # Disable Chrome sandbox to prevent Chrome crashes due to restricted security setup 
    chrome_options.add_argument("--no-sandbox")  
    # Disable Chrome shared memory (uses temporary storage instead) to prevent crashes due to limited ressources in Docker containers
    chrome_options.add_argument("--disable-dev-shm-usage")  
    # chrome_options.add_argument("--headless")  # run Chrome without opening a Browser window
    # Create a Chrome webdriver with custom options 
    driver = webdriver.Chrome(options=chrome_options)

    try:
        # Get request to frontend Gradio UI  
        # Make sure the Docker container is running locally and port 7860 is mapped
        driver.get("http://localhost:7860")

        # Make inputs in Gradio UI
        make_number_input(driver, "Age", 30)
        make_dropdown_input(driver, "Married/Single", "Single")
        make_number_input(driver, "Income", 300000)
        make_dropdown_input(driver, "Car Ownership", "No")
        make_dropdown_input(driver, "House Ownership", "Neither Rented Nor Owned")
        make_slider_input(driver, "Current House Years", 11)
        make_dropdown_input(driver, "City", "Sikar")
        make_dropdown_input(driver, "State", "Rajasthan")
        make_dropdown_input(driver, "Profession", "Artist")
        make_slider_input(driver, "Experience", 3)
        make_slider_input(driver, "Current Job Years", 3)

        # Click predict button
        predict_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "predict-button")))
        predict_button.click()

        # --- Prediction result ---
        # Find probability elements
        default_probability = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, "//dl[contains(@class, 'label')]//dt[text()='Default']/following-sibling::dd")))
        no_default_probability = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, "//dl[contains(@class, 'label')]//dt[text()='No Default']/following-sibling::dd")))
        # Extract numbers
        default_probability = int(default_probability.text.replace("%", ""))
        no_default_probability = int(no_default_probability.text.replace("%", ""))
        # Find prediction text element
        prediction = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//div[@id='prediction-text']//textarea")))
        # Extract prediction text 
        prediction_text = prediction.get_attribute("value")
        
        # --- Assert ---
        # Ensure prediction is as expected
        assert prediction_text in ["Default", "No Default"]
        # Ensure probabilities are numbers between 0 and 100
        assert 0 <= default_probability <=100
        assert 0 <= no_default_probability <= 100
        # Ensure probabilities sum to approximately 100
        sum = default_probability + no_default_probability
        assert 99 <= sum <= 101  # allow for rounding edge cases
         
    finally:
        time.sleep(5)  # remove after dev/test phase
        # Close Chrome browser window
        driver.quit()


# End-to-end test that simulates a user submitting out-of-range values and receiving an error message in the frontend UI 
@pytest.mark.e2e
def test_user_submits_out_of_range_values():
    pass


# End-to-end test that simulates a user submitting a form with missing required fields and receiving an error message in the frontend UI 
@pytest.mark.e2e
def test_user_submits_form_with_empty_required_fields():
    pass