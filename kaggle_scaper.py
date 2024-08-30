import os
from pprint import pprint

# from langchain_openai import ChatOpenAI
import httpx
from langchain_openai import ChatOpenAI
from pymongo import MongoClient
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# from webdriver_manager.chrome import ChromeDriverManager
from markdownify import markdownify
from dotenv import load_dotenv

from langchain.pydantic_v1 import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

from prompts.summarizer_prompt import (
    CHALLENGE_DATA_PROMPT,
    CHALLENGE_DESCRIPTOIN_PROMPT,
    CHALLENGE_EVALUATION_PROMPT,
)
from states.main import KaggleProblemState


class Evaluation(BaseModel):
    description: str = Field(description="usefull description for evaluating data")
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
        Initializes the KaggleDataUtils with configuration and proxy settings.

        Args:
            config (dict): Configuration settings for the utility.
            proxy (httpx.Client): HTTP client for proxy settings.
        """
        # proxy = httpx.Client(proxy="http://127.0.0.1:2080")
        self.scraped_data_collection = client["challenge_data"].get_collection(
            "scraped_data"
        )
        self.config = config
        self.mongo_dict = {}
        self.llm = ChatOpenAI(
            base_url=base_url,
            model="gpt-4o-mini",
            http_client=proxy,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
        )

    def extract_challenge_details(self, challenge_url: str):
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
        if not challenge_url.endswith("/"):
            challenge_url += "/"
        # if os.getenv("HTTP_PROXY_URL", None):
        chrome_options.add_argument(f"--proxy-server=http://127.0.0.1:2080")
        chrome_options.add_argument("--blink-settings=imagesEnabled=false")
        # chrome_options.add_argument("--headless=new")
        driver = webdriver.Chrome(
            # service=
            options=chrome_options
            # service=Service("/home/msnp/chromedriver_linux64/")
            # )
        )

        driver.get(challenge_url + "overview/")

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

        driver.get(challenge_url + "data/")
        driver.implicitly_wait(5)

        # Wait for the challenge details to load
        wait = WebDriverWait(driver, 35)

        wait.until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )

        challenge_data_details = driver.find_element(
            By.XPATH,
            "/html/body/main/div[1]/div/div[5]/div[2]/div/div/div[6]/div[1]/div[1]/div/div[2]/div",
        ).get_attribute("innerHTML")

        driver.quit()
        d = {
            "description": markdownify(challenge_description),
            "evaluation": markdownify(challenge_evaluation),
            "data_details": markdownify(challenge_data_details),
        }
        self.mongo_dict.update({"challenge_url": challenge_url, "scraped_data": d})
        for i, j in d.items():
            with open(f"./kaggle_challenges_data/{i}.md", "w") as f:
                f.write(j)
        return d

    def get_saved_challenge_data(self, challenge_url):
        result = self.scraped_data_collection.find_one({"challenge_url": challenge_url})
        return result

    def summarize_data(self, description, evaluation, data_details):
        eval_parser = PydanticOutputParser(pydantic_object=Evaluation)
        data_parser = PydanticOutputParser(pydantic_object=Evaluation)

        desc_inp = ChatPromptTemplate.from_messages(
            CHALLENGE_DESCRIPTOIN_PROMPT
        ).format(text=description)
        eval_inp = ChatPromptTemplate.from_messages(CHALLENGE_EVALUATION_PROMPT).format(
            text=evaluation, format_instructions=eval_parser.get_format_instructions()
        )
        data_inp = ChatPromptTemplate.from_messages(CHALLENGE_DATA_PROMPT).format(
            text=data_details
        )
        results = self.llm.batch([desc_inp, eval_inp, data_inp])
        self.mongo_dict.update(
            {
                "summarized": {
                    "description": results[0].content,
                    "evaluation": eval_parser.invoke(results[1]).dict(),
                    "data_details": results[2].content,
                }
            }
        )
        self.scraped_data_collection.insert_one(self.mongo_dict)
        return {
            "description": results[0].content,
            "evaluation": eval_parser.invoke(results[1]),
            "data_details": results[2].content,
        }

    def __call__(self, state: KaggleProblemState):
        challenge_url = state.challenge_url
        result = self.get_saved_challenge_data(challenge_url)
        data = None
        if result:
            data = result["summarized"]
        else:
            dict_ = self.extract_challenge_details(challenge_url)
            data = self.summarize_data(**dict_)

        return {
            "index": -1,
            "problem_description": data["description"],
            "dataset_info": data["data_details"],
            "evaluation_metric": data["evaluation"]["metric"],
        }


if __name__ == "__main__":
    load_dotenv(override=True)
    client = MongoClient(
        host=os.getenv("MONGO_HOST"), port=int(os.getenv("MONGO_PORT"))
    )
    s = ScrapeKaggle(client, proxy=None)
    s(
        KaggleProblemState(
            **{
                "challenge_url": "https://www.kaggle.com/competitions/nlp-getting-started/"
            }
        )
    )
