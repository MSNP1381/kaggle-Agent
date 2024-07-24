
import json
from typing import Any, Dict
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from prompts.utils import DATASET_ANALYSIS_PROMPT
from states.main import KaggleProblemState


class KaggleDataUtils:
    def __init__(self, config, proxy):
        """
        Initializes the KaggleDataUtils with configuration and proxy settings.

        Args:
            config (dict): Configuration settings for the utility.
            proxy (httpx.Client): HTTP client for proxy settings.
        """
        self.config = config
        self.llm = ChatOpenAI(model="gpt-4o-mini", http_client=proxy, temperature=0)
        self.dataset_analysis_prompt = ChatPromptTemplate.from_template(DATASET_ANALYSIS_PROMPT)

    def analyze_dataset(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes the dataset and provides a description and preprocessing recommendations.

        Args:
            dataset (pd.DataFrame): The dataset to analyze.

        Returns:
            Dict[str, Any]: A dictionary containing the description and preprocessing steps.
        """
        try:
            dataset_overview = self._generate_dataset_overview(dataset)
            response = self.llm.invoke(self.dataset_analysis_prompt.format_messages(
                dataset_overview=dataset_overview
            ), config=self.config)

            result = self._parse_response(response.content)
            return result
        except Exception as e:
            print(f"An error occurred while analyzing the dataset: {e}")
            return {}

    def _generate_dataset_overview(self, dataset: pd.DataFrame) -> str:
        """
        Generates an overview of the dataset.

        Args:
            dataset (pd.DataFrame): The dataset to generate an overview for.

        Returns:
            str: A string representation of the dataset overview.
        """
        overview = [f"Shape: {dataset.shape}", f"Columns: {dataset.columns.tolist()}", f"Data Types:\n{dataset.dtypes}",
                    f"Missing Values:\n{dataset.isnull().sum()}", f"Unique Values:\n{dataset.nunique()}"]
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
        preprocessing_steps = lines[-1].split(";")

        return {
            "description": description,
            "preprocessing_steps": preprocessing_steps
        }

    def __call__(self, state: KaggleProblemState):

        result = self.analyze_dataset(pd.read_csv(state.dataset_path))

        return {'dataset_info': json.dumps(result, indent=1)}

