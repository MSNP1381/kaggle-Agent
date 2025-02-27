import json
import os
import pprint
import unittest
import zipfile
import logging
from unittest.mock import MagicMock

from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.rest import ApiException
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.fix import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from markdownify import markdownify
from pydantic import BaseModel, Field
from pymongo import MongoClient
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.proxy import Proxy, ProxyType
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from prompts.summarizer_prompt import (
    CHALLENGE_DATA_PROMPT,
    CHALLENGE_DESCRIPTION_PROMPT,
    CHALLENGE_EVALUATION_PROMPT,
)
from states.main import KaggleProblemState
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
    ):
        """
        Initializes the DataUtils with configuration and proxy settings.

        Args:
            config (dict): Configuration settings for the utility.
            proxy (str): Proxy URL for HTTP requests.
        """
        self.first_time_scrape = False
        self.scraped_data_collection = client["challenge_data"].get_collection(
            "scraped_data"
        )
        self.config = config
        self.mongo_dict = {}
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
        )

        self.dataset_collection = client["challenge_data"].get_collection("datasets")

        # Initialize Kaggle API
        self.kaggle_api = KaggleApi()
        self.kaggle_api.authenticate()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("ScrapeKaggle initialized")

    def extract_challenge_details(self, challenge_url):
        """
        Extracts challenge details from the given URL.

        Args:
            challenge_url (str): The URL of the challenge.

        Returns:
            dict: A dictionary containing the challenge details.
        """
        self.logger.info(f"Extracting challenge details for URL: {challenge_url}")
        data = self.scraped_data_collection.find_one({"challenge_url": challenge_url})
        if data:
            self.logger.info("Data found in database")
            d = {
                "description": data["scraped_data"]["description"],
                "evaluation": data["scraped_data"]["evaluation"],
                "data_details": data["scraped_data"]["data_details"],
            }
            self.mongo_dict.update({"scraped_data": d})
            return d

        self.logger.info("Data not found in database, scraping from web")
        options = webdriver.ChromeOptions()
        # options.add_argument("--headless")
        proxy = os.getenv("HTTP_PROXY")
        print(proxy)
        p = Proxy()
        p.proxy_type = ProxyType.MANUAL
        p.http_proxy = proxy
        p.httpProxy = proxy
        p.ssl_proxy = proxy

        options.proxy = p
        if proxy:
            options.add_argument(f"--proxy-server={proxy}")

        driver = webdriver.Chrome(
            # command_executor="http://127.0.0.1:4444",
            options=options
        )

        driver.get(append_url(challenge_url, "overview"))

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

        driver.get(append_url(challenge_url, "data"))
        # Wait for the challenge details to load
        wait = WebDriverWait(driver, 35)
        wait.until(
            EC.presence_of_element_located(
                (
                    By.CSS_SELECTOR,
                    "#site-content > div:nth-child(2) > div > div > div.sc-kGGICJ.gdXvLH > div.sc-fOkpiL.gnTWUc > div.sc-sYGvw.jspRTt > div > div:nth-child(2) > div > div.sc-eJqIHI > div > div > div",
                )
            )
        )

        challenge_data_details = driver.find_element(
            By.CSS_SELECTOR,
            "#site-content > div:nth-child(2) > div > div > div.sc-kGGICJ.gdXvLH > div.sc-fOkpiL.gnTWUc > div.sc-sYGvw.jspRTt > div > div:nth-child(2) > div > div.sc-eJqIHI.eRnvzm > div > div > div",
        ).get_attribute("innerHTML")

        d = {
            "description": markdownify(challenge_description),
            "evaluation": markdownify(challenge_evaluation),
            "data_details": markdownify(challenge_data_details),
        }
        if not challenge_url.endswith("/"):
            challenge_url += "/"
        self.mongo_dict.update({"challenge_url": challenge_url, "scraped_data": d})
        for i, j in d.items():
            with open(f"./kaggle_challenges_data/{i}.md", "w", encoding="utf-8") as f:
                f.write(j)
        driver.quit()
        self.logger.info("Challenge details extracted successfully")
        return d

    def get_saved_challenge_data(self, challenge_url):
        """
        Retrieves saved challenge data from the database.

        Args:
            challenge_url (str): The URL of the challenge.

        Returns:
            dict: A dictionary containing the saved challenge data, or None if not found.
        """
        self.logger.info(f"Retrieving saved challenge data for URL: {challenge_url}")
        if not challenge_url.endswith("/"):
            challenge_url += "/"
        return self.scraped_data_collection.find_one({"challenge_url": challenge_url})

    def _init_state(
        self,
    ):
        self.logger.info("Initializing state")
        self.dataset_path = "./input/train.csv"
        self.test_dataset_path = "./input/test.csv"

        return {
            # "file_env_var": env_var,
            "dataset_path": self.dataset_path,
            "test_dataset_path": self.test_dataset_path,
        }

    def __call__(self, state: KaggleProblemState):
        self.logger.info(f"Processing state for challenge URL: {state.challenge_url}")
        challenge_url = state.challenge_url
        self.mongo_dict["challenge_url"] = challenge_url
        download_info = self.download_challenge_data(challenge_url)
        result: dict = self.get_saved_challenge_data(challenge_url)
        data = None
        if result:
            if result.get("scraped_data") and not result.get("summarized"):
                dict_ = result["scraped_data"]
                data = self.summarize_data(**dict_)
            elif result.get("summarized"):
                data = result["summarized"]
            else:
                raise ValueError("No scraped data or summarized data found")
        else:
            self.first_time_scrape = True
            dict_ = self.extract_challenge_details(challenge_url)
            data = self.summarize_data(**dict_)

        # Download challenge data and leaderboard
        base_response = {
            "index": -1,
            "problem_description": data["description"],
            "dataset_info": data["data_details"]
            + "\n\n\n---\n trian dataset path: ~/train.csv \n\n\n---\n test dataset path: ~/test.csv",
            "evaluation_metric": data["evaluation"]["metric"],
            "evaluation_description": data["evaluation"]["description"],
            "data_path": download_info["data_path"] if download_info else None,
            "leaderboard_path": (
                download_info["leaderboard_path"] if download_info else None
            ),
        }
        d = self._init_state()
        self.logger.info("State processed successfully")
        return base_response | d

    def download_challenge_data(self, challenge_url):
        """
        Downloads the challenge data from Kaggle and fetches the leaderboard.

        Args:
            challenge_url (str): The URL of the challenge.

        Returns:
            dict: A dictionary containing paths to the downloaded data and leaderboard info.
        """
        self.logger.info(f"Downloading challenge data for URL: {challenge_url}")
        ls_dir = os.listdir("./input")
        if ls_dir:
            return {
                "data_path": "./input",
                "leaderboard_path": "./input/leaderboard.json",
            }
        # Extract competition name from URL
        competition_name = challenge_url.split("/")[-1]
        print(
            "*" * 20,
            "\n" * 3,
            competition_name,
            challenge_url,
            "\n" * 3,
            "*" * 20,
            "\n" * 3,
        )
        try:
            # Create directories to store the downloaded files
            ongoing_dir = "./input"
            os.makedirs(ongoing_dir, exist_ok=True)

            # Download competition files
            self.kaggle_api.competition_download_files(
                competition_name, path=ongoing_dir
            )
            # Decompress all .zip files in the ongoing directory
            for file_name in os.listdir(ongoing_dir):
                if file_name.endswith(".zip"):
                    file_path = os.path.join(ongoing_dir, file_name)
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        zip_ref.extractall(ongoing_dir)
                    print(f"Decompressed {file_name}")
            self.logger.info(f"Successfully downloaded data for {competition_name}")

            # Fetch leaderboard
            leaderboard = self.kaggle_api.competition_view_leaderboard(competition_name)
            leaderboard_path = os.path.join(ongoing_dir, "leaderboard.json")
            with open(leaderboard_path, "w") as f:
                json.dump(leaderboard, f, indent=2)

            self.logger.info(f"Successfully fetched leaderboard for {competition_name}")

            return {"data_path": ongoing_dir, "leaderboard_path": leaderboard_path}

        except ApiException as e:
            self.logger.error(f"Error processing data for {competition_name}: {str(e)}")
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
        self.logger.info("Summarizing challenge data")
        description_prompt = ChatPromptTemplate.from_messages(
            CHALLENGE_DESCRIPTION_PROMPT, "mustache"
        )
        evaluation_prompt = ChatPromptTemplate.from_messages(
            CHALLENGE_EVALUATION_PROMPT, "mustache"
        )
        data_prompt = ChatPromptTemplate.from_messages(
            CHALLENGE_DATA_PROMPT, "mustache"
        )
        parser = PydanticOutputParser(pydantic_object=Evaluation)
        description_chain = description_prompt | self.llm | StrOutputParser()
        evaluation_chain = (
            evaluation_prompt
            | self.llm
            | PydanticOutputParser(pydantic_object=Evaluation)
        )
        data_chain = data_prompt | self.llm | StrOutputParser()

        summarized_description = description_chain.invoke({"text": description})
        summarized_evaluation = evaluation_chain.invoke(
            {
                "text": evaluation,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        summarized_data = data_chain.invoke({"text": data_details})

        result = {
            "description": summarized_description,
            "evaluation": summarized_evaluation.model_dump(),
            "data_details": summarized_data,
        }

        self.mongo_dict.update({"summarized": result})
        if self.first_time_scrape:
            self.scraped_data_collection.insert_one(self.mongo_dict)
        else:
            self.scraped_data_collection.update_one(
                {"challenge_url": self.mongo_dict["challenge_url"]},
                {
                    "$set": {
                        "summarized": result,
                    }
                },
            )
        self.logger.info("Challenge data summarized successfully")
        return result


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

    load_dotenv(override=True)
    client = MongoClient(
        host=os.getenv("MONGO_HOST"), port=int(os.getenv("MONGO_PORT"))
    )
    state = KaggleProblemState(
        challenge_url="https://www.kaggle.com/c/spaceship-titanic"
    )
    scraper = ScrapeKaggle(
        client,
    )
    pprint.pprint(scraper(state))
