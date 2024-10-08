import os
import json
import pprint
import zipfile
import httpx
from langchain_openai import ChatOpenAI
from pymongo import MongoClient
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.fix import StrOutputParser
from kaggle.api.kaggle_api_extended import KaggleApi
from markdownify import markdownify
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

from prompts.summarizer_prompt import (
    CHALLENGE_DATA_PROMPT,
    CHALLENGE_DESCRIPTION_PROMPT,
    CHALLENGE_EVALUATION_PROMPT,
)
from states.main import KaggleProblemState

import requests
from bs4 import BeautifulSoup
from kaggle.rest import ApiException


from utils import append_url


class Evaluation(BaseModel):
    description: str = Field(description="useful description for evaluating data")
    metric: str = Field(
        description="metric for evaluating challenge like f1 score,accuracy,precision, etc. which is mentioned in text"
    )



class ScrapeKaggle:
    def __init__(
        self,
        client: MongoClient,
        config=None,
        proxy=None,
        base_url="https://api.avalai.ir/v1",
    ):
        """
        Initializes the DataUtils with configuration and proxy settings.

        Args:
            config (dict): Configuration settings for the utility.
            proxy (str): Proxy URL for HTTP requests.
        """
        self.scraped_data_collection = client["challenge_data"].get_collection(
            "scraped_data"
        )
        self.config = config
        self.mongo_dict = {}
        http_client=None
        if proxy:
           http_client= httpx.Client(proxy=proxy)
        self.llm = ChatOpenAI(
                base_url=base_url,
                model="gpt-4o-mini",
                http_client=http_client,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0,
            )

        self.dataset_collection = client["challenge_data"].get_collection("datasets")

        # Initialize Kaggle API
        self.kaggle_api = KaggleApi()
        self.kaggle_api.authenticate()

    def extract_challenge_details(self, challenge_url):
        """
        Extracts challenge details from the given URL.

        Args:
            challenge_url (str): The URL of the challenge.

        Returns:
            dict: A dictionary containing the challenge details.
        """
        # chrome_options.add_argument("--blink-settings=imagesEnabled=false")
        # chrome_options.add_argument("--disable-gpu")
        # chrome_options.add_argument("--headless")
        # chrome_options.add_argument("--headless=new")
        
        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--no-sandbox')
            options.add_argument('--headless') 
            driver = webdriver.Remote(
                command_executor='http://localhost:4444/wd/hub',
                options=options
            )
   

            driver.get(append_url(challenge_url , "overview"))

            # Wait for the challenge details to load
            wait = WebDriverWait(driver, 35)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#Description")))

            # Extract the challenge details
            challenge_description = driver.find_element(
                By.CSS_SELECTOR, "#description > div > div:nth-child(2)"
            ).get_attribute("innerHTML")

            challenge_evaluation = driver.find_element(
                By.CSS_SELECTOR, "#evaluation > div > div:nth-child(2)"
            ).get_attribute("innerHTML")

            driver.get(append_url(challenge_url , "data"))
            driver.implicitly_wait(5)

            # Wait for the challenge details to load
            wait = WebDriverWait(driver, 35)
            wait.until(
                EC.presence_of_element_located((By.TAG_NAME, "h3"))

                # lambda d: d.execute_script("return document.readyState") == "complete"
            )

            challenge_data_details = driver.find_element(
                By.XPATH,
                "/html/body/main/div[1]/div/div[5]/div[2]/div/div/div[6]/div[1]/div[1]/div/div[2]/div",
            ).get_attribute("innerHTML")

            d = {
                "description": markdownify(challenge_description),
                "evaluation": markdownify(challenge_evaluation),
                "data_details": markdownify(challenge_data_details),
            }
            if not challenge_url.endswith("/"):
                challenge_url+='/'
            self.mongo_dict.update({"challenge_url": challenge_url, "scraped_data": d})
            for i, j in d.items():
                with open(f"./kaggle_challenges_data/{i}.md", "w") as f:
                    f.write(j)
            driver.quit()
            return d
        except Exception as e :
            print(e)
            raise e

    def get_saved_challenge_data(self, challenge_url):
        """
        Retrieves saved challenge data from the database.

        Args:
            challenge_url (str): The URL of the challenge.

        Returns:
            dict: A dictionary containing the saved challenge data, or None if not found.
        """
        if not challenge_url.endswith("/"):
            challenge_url+='/'
        return self.scraped_data_collection.find_one({"challenge_url": challenge_url})
    
    def _init_state(self, ):
        self.dataset_path = "./ongoing/train.csv"
        self.test_dataset_path = "./ongoing/test.csv"
        # with open(self.dataset_path) as f:
        #     env_var = self.nb_executor.upload_file_env(f)
        # with open(self.test_dataset_path) as f:
        #     env_var = self.nb_executor.upload_file_env(f, env_var="TEST_FILE")

        
        return{
                # "file_env_var": env_var,
                "dataset_path": self.dataset_path,
                "test_dataset_path": self.test_dataset_path,
            }
        
    def __call__(self, state: KaggleProblemState):
        challenge_url = state.challenge_url
        download_info = self.download_challenge_data(challenge_url)
        result = self.get_saved_challenge_data(challenge_url)
        data = None
        if result:
            data = result["summarized"]
        else:
            dict_ = self.extract_challenge_details(challenge_url)
            data = self.summarize_data(**dict_)

        # Download challenge data and leaderboard
        base_response= {
            "index": -1,
            "problem_description": data["description"],
            "dataset_info": data["data_details"]+"\n\n\n---\n trian dataset path: ~/train.csv \n\n\n---\n test dataset path: ~/test.csv",
            "evaluation_metric": data["evaluation"]["metric"],
            "evaluation_description": data["evaluation"]["description"],
            "data_path": download_info["data_path"] if download_info else None,
            "leaderboard_path": download_info["leaderboard_path"] if download_info else None
        }
        d=self._init_state()
        return base_response | d

    def download_challenge_data(self, challenge_url):
        """
        Downloads the challenge data from Kaggle and fetches the leaderboard.

        Args:
            challenge_url (str): The URL of the challenge.

        Returns:
            dict: A dictionary containing paths to the downloaded data and leaderboard info.
        """
        # Extract competition name from URL
        competition_name = challenge_url.split('/')[-1]
        print("*"*20,"\n"*3,competition_name,challenge_url,"*"*20,"\n"*3)
        try:
            # Create directories to store the downloaded files
            ongoing_dir = f"./ongoing"
            os.makedirs(ongoing_dir, exist_ok=True)

            # Download competition files
            self.kaggle_api.competition_download_files(
                competition_name,
                path=ongoing_dir
            )
            # Decompress all .zip files in the ongoing directory
            for file_name in os.listdir(ongoing_dir):
                if file_name.endswith(".zip"):
                    file_path = os.path.join(ongoing_dir, file_name)
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(ongoing_dir)
                    print(f"Decompressed {file_name}")
            print(f"Successfully downloaded data for {competition_name}")

            # Fetch leaderboard
            leaderboard = self.kaggle_api.competition_view_leaderboard(competition_name)
            leaderboard_path = os.path.join(ongoing_dir, "leaderboard.json")
            with open(leaderboard_path, 'w') as f:
                json.dump(leaderboard, f, indent=2)

            print(f"Successfully fetched leaderboard for {competition_name}")

            return {
                "data_path": ongoing_dir,
                "leaderboard_path": leaderboard_path
            }

        except ApiException as e:
            print(f"Error processing data for {competition_name}: {str(e)}")
            return None

    def summarize_data(self, description, evaluation, data_details):
        """
        Summarizes the challenge data using the LLM.

        Args:
            description (str): Challenge description.
            evaluation (str): Evaluation criteria.
            data_details (str): Data details.

        Returns:
            dict: A dictionary containing summarized challenge information.
        """
        description_prompt = ChatPromptTemplate.from_messages(CHALLENGE_DESCRIPTION_PROMPT,'mustache')
        evaluation_prompt = ChatPromptTemplate.from_messages(CHALLENGE_EVALUATION_PROMPT,'mustache')
        data_prompt = ChatPromptTemplate.from_messages(CHALLENGE_DATA_PROMPT,'mustache')
        parser=PydanticOutputParser(pydantic_object=Evaluation)
        description_chain = description_prompt | self.llm |StrOutputParser()
        evaluation_chain = evaluation_prompt | self.llm | PydanticOutputParser(pydantic_object=Evaluation)
        data_chain = data_prompt | self.llm|StrOutputParser()

        summarized_description = description_chain.invoke({"text": description})
        summarized_evaluation = evaluation_chain.invoke({"text": evaluation,'format_instructions':parser.get_format_instructions()})
        summarized_data = data_chain.invoke({"text": data_details})

        result = {
            "description": summarized_description,
            "evaluation": summarized_evaluation.model_dump(),
            "data_details": summarized_data,
        }

        self.mongo_dict.update({"summarized": result})
        self.scraped_data_collection.insert_one(self.mongo_dict)

        return result

import unittest
from unittest.mock import MagicMock

class TestScrapeKaggle(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.scraper = ScrapeKaggle(self.client, proxy="http://127.0.0.1:2080")

    def test_scrape_kaggle(self):
        challenge_url = "https://www.kaggle.com/competitions/titanic"
        state = KaggleProblemState(challenge_url=challenge_url)
        result = self.scraper(state)
        self.assertIsNotNone(result)
        self.assertIn("index", result)
        self.assertIn("problem_description", result)
        self.assertIn("dataset_info", result)
        self.assertIn("evaluation_metric", result)
        self.assertIn("evaluation_description", result)
        self.assertIn("data_path", result)
        self.assertIn("leaderboard_path", result)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    client=MongoClient(
            host=os.getenv("MONGO_HOST"), port=int(os.getenv("MONGO_PORT"))
        )
    state=KaggleProblemState(challenge_url="https://www.kaggle.com/competitions/titanic")
    scraper = ScrapeKaggle(client,)
    pprint.pprint(scraper(state))
