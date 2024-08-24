from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from markdownify import markdownify
import argparse


# from  html2markdown import  convert as markdownify
def extract_challenge_details(challenge_url, proxy_host=None, proxy_port=None):
    """
    Extracts detailed information about a Kaggle challenge from its URL.

    Args:
        challenge_url (str): URL of the Kaggle challenge.
        proxy_host (str, optional): Proxy server host.
        proxy_port (int, optional): Proxy server port.

    Returns:
        dict: A dictionary containing the challenge details.
    """
    chrome_options = Options()
    if proxy_host and proxy_port:
        chrome_options.add_argument(f"--proxy-server={proxy_host}:{proxy_port}")
    chrome_options.add_argument("--blink-settings=imagesEnabled=false")

    driver = webdriver.Chrome(options=chrome_options)
    driver.get(challenge_url + "overview/")

    # Wait for the challenge details to load
    wait = WebDriverWait(driver, 15)
    challenge_details = wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "#Description"))
    )

    # Extract the challenge details
    challenge_description = driver.find_element(
        By.CSS_SELECTOR, "#description > div > div:nth-child(2)"
    ).get_attribute("innerHTML")

    challenge_evaluation = driver.find_element(
        By.CSS_SELECTOR, "#evaluation > div > div:nth-child(2)"
    ).get_attribute("innerHTML")

    driver.get(challenge_url + "data/")

    # Wait for the challenge details to load
    wait = WebDriverWait(driver, 15)
    challenge_details = wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".sc-cjvFHf"))
    )

    challenge_data_details = driver.find_element(
        By.XPATH,
        "/html/body/main/div[1]/div/div[5]/div[2]/div/div/div[6]/div[1]/div[1]/div/div[2]/div",
    ).get_attribute("innerHTML")
    # challenge_data_details
    # challenge_deadline = driver.find_element(By.CSS_SELECTOR, ".competition-detail .deadline").text
    # challenge_prize = driver.find_element(By.CSS_SELECTOR, ".competition-detail .prize").text
    # challenge_participants = driver.find_element(By.CSS_SELECTOR, ".competition-detail .participants").text
    # challenge_data_description = driver.find_element(By.CSS_SELECTOR, ".data-description").text

    # Close the webdriver
    with open("xxx.html", "w") as f:
        f.write(challenge_data_details)
    driver.quit()

    return {
        "description": markdownify(challenge_description),
        "evaluation": markdownify(challenge_evaluation),
        "data_details": markdownify(challenge_data_details),
    }


parser = argparse.ArgumentParser("kaggle_scraper")
parser.add_argument(
    "--url",
    help="url to challenge",
    type=str,
    required=True,
)
parser.add_argument(
    "--proxy_url",
    required=False,
    help="url to proxy",
    default="http://127.0.0.1",
    type=str,
)
parser.add_argument(
    "--port",
    required=False,
    help="proxy port",
    default=2080,
    type=int,
)

args = parser.parse_args()


# Usage example
d = extract_challenge_details(
    args.url,
    proxy_host=args.proxy_url,
    proxy_port=args.port,
)

for i, j in d.items():
    with open(f"./kaggle_challenges_data/{i}.md", "w") as f:
        f.write(j)
