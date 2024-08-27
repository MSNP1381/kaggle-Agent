import json
import os
from typing import Any, Dict
from dotenv import load_dotenv
import httpx
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from prompts.utils import DATASET_ANALYSIS_PROMPT
from states.main import KaggleProblemState
from typing import Dict, List, Optional


class DatUtilState(BaseModel):
    """
    A Pydantic model representing the state of a dataset utility for machine learning problems.

    This class encapsulates various aspects of dataset analysis, including qualitative and
    quantitative descriptions, preprocessing recommendations, and feature selection insights.
    """

    dataset_overview: str = Field(
        description="A brief overview of the dataset, including its purpose and context"
    )

    qualitative_description: str = Field(
        description="A detailed qualitative description of the dataset's features and characteristics"
    )

    quantitative_description: str = Field(
        description="Quantitative details about the dataset, including data types, missing values, and unique values"
    )

    insights: Optional[str] = Field(
        default=None,
        description="Any  insights or recommendations from the dataset analysis",
    )


class KaggleDataUtils:
    def __init__(self, config, proxy, base_url="https://api.avalai.ir/v1"):
        """
        Initializes the KaggleDataUtils with configuration and proxy settings.

        Args:
            config (dict): Configuration settings for the utility.
            proxy (httpx.Client): HTTP client for proxy settings.
        """
        self.config = config
        self.llm = ChatOpenAI(
            base_url=base_url, model="gpt-4o-mini", http_client=proxy, temperature=0
        )
        self.dataset_analysis_prompt = ChatPromptTemplate.from_template(
            DATASET_ANALYSIS_PROMPT
        )
        self.output_parser = PydanticOutputParser(pydantic_object=DatUtilState)
        self.chain = self.dataset_analysis_prompt | self.llm | self.output_parser

    def analyze_dataset(self, dataset: pd.DataFrame, data_initial_info):
        """
        Analyzes the dataset and provides a description and preprocessing recommendations.

        Args:
            dataset (pd.DataFrame): The dataset to analyze.

        Returns:
            Dict[str, Any]: A dictionary containing the description and preprocessing steps.
        """
        try:
            dataset_overview = self._generate_dataset_overview(dataset)
            dataset_head = dataset.head().to_markdown(index=False)
            response = self.chain.invoke(
                {
                    "dataset_overview": dataset_overview,
                    "dataset_head": dataset_head,
                    "data_initial_info": data_initial_info,
                    "format_instructions": self.output_parser.get_format_instructions(),
                },
                config=self.config,
            )

            return response
        except Exception as e:
            print(f"An error occurred while analyzing the dataset: {e}")
            return None

    def _generate_dataset_overview(self, dataset: pd.DataFrame) -> str:
        """
        Generates an overview of the dataset.

        Args:
            dataset (pd.DataFrame): The dataset to generate an overview for.

        Returns:
            str: A string representation of the dataset overview.
        """
        overview = [
            f"Shape: {dataset.shape}",
            f"Columns: {dataset.columns.tolist()}",
            f"Data Types:\n{dataset.dtypes}",
            f"Missing Values:\n{dataset.isnull().sum()}",
            f"Unique Values:\n{dataset.nunique()}",
        ]
        return "\n".join(overview)

    def _parse_response(self, response_content: str) -> Dict[str, Any]:
        """
        Parses the response from the LLM.

        Args:
            response_content (str): The response content from the LLM.

        Returns:
            Dict[str, Any]: A dictionary containing the description and preprocessing steps.
        """
        # Here, we assume the response is structured in a way that we can directly parse it.
        # You may need to adjust this based on the actual response format.
        lines = response_content.strip().split("\n")
        description = "\n".join(lines[:-1])

        return {
            "description": description,
        }

    def __call__(self, state: KaggleProblemState):

        result = self.analyze_dataset(
            pd.read_csv(state.dataset_path), state.dataset_info
        )
        # print(result)

        return {"dataset_info": result.json().replace("\\n", "\n")}


if __name__ == "__main__":
    load_dotenv()
    proxy = httpx.Client(proxy=os.getenv("HTTP_PROXY_URL"))
    data_util = KaggleDataUtils(None, proxy)
    dataset_path = "./house_prices.csv"
    problem_description = f"""
    Predict house prices based on various features.
    The evaluation metric is Root Mean Squared Error (RMSE).
    The dataset contains information about house features and their corresponding sale prices.
    dataset file name is : "{dataset_path}"
    """

    print(
        data_util(
            KaggleProblemState(
                **{
                    "index": -1,
                    "problem_description": problem_description,
                    "dataset_path": dataset_path,
                    "evaluation_metric": "r2_score",
                }
            )
        )
    )
