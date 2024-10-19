import os
from typing import Any, Dict, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from pymongo import MongoClient

from prompts.utils import DATASET_ANALYSIS_PROMPT
from states.main import KaggleProblemState


class DatasetAnalysis(BaseModel):
    quantitative_analysis: str = Field(
        description="Detailed quantitative analysis of the dataset"
    )
    qualitative_analysis: str = Field(
        description="Detailed qualitative analysis of the dataset"
    )


class DataUtils:
    def __init__(
        self,
        config: Dict[str, Any],
        llm: ChatOpenAI,
        mongo_client: Optional[MongoClient] = None,
    ):
        self.config = config
        self.llm = llm
        self.mongo_client = mongo_client
        self.dataset_analysis_prompt = ChatPromptTemplate.from_messages(
            [("system", DATASET_ANALYSIS_PROMPT)]
        )
        self.output_parser = PydanticOutputParser(pydantic_object=DatasetAnalysis)

    def analyze_dataset(
        self, dataset: pd.DataFrame, dataset_info: str
    ) -> DatasetAnalysis:
        data_initial_info = self._generate_dataset_overview(dataset)
        dataset_head = dataset.head().to_markdown()

        format_instructions = self.output_parser.get_format_instructions()
        response = (
            self.dataset_analysis_prompt | self.llm | self.output_parser
        ).invoke(
            {
                "data_initial_info": data_initial_info,
                "dataset_overview": dataset_info,
                "dataset_head": dataset_head,
                "format_instructions": format_instructions,
            },
            config=self.config,
        )
        return response

    def _generate_dataset_overview(self, dataset: pd.DataFrame) -> str:
        overview = [
            f"Shape: {dataset.shape}",
            f"Columns: {dataset.columns.tolist()}",
            f"Data Types:\n{dataset.dtypes}",
            f"Missing Values:\n{dataset.isnull().sum()}",
            f"Unique Values:\n{dataset.nunique()}",
            f"Numerical Columns Summary:\n{dataset.describe().to_string()}",
        ]
        return "\n".join(overview)

    def __call__(self, state: KaggleProblemState) -> Dict[str, str]:
        if self.mongo_client:
            db = self.mongo_client.get_database("challenge_data")
            collection = db.get_collection("data_utils_results")
            if not state.challenge_url.endswith("/"):
                state.challenge_url += "/"
            data = collection.find_one({"challenge_url": state.challenge_url})
            if data:
                return {k: v for k, v in data.items() if k != "_id"}

        dataset = self._load_dataset(state.dataset_path)
        if dataset is None:
            return {"error": "Failed to load dataset"}
        result = self.analyze_dataset(dataset, state.problem_description)
        analysis_result = self._generate_dataset_overview(dataset)

        output = {
            "challenge_url": state.challenge_url,
            "dataset_info": analysis_result,
            "quantitative_analysis": result.quantitative_analysis,
            "qualitative_analysis": result.qualitative_analysis,
        }

        # Write result back to MongoDB if available
        if self.mongo_client:
            try:
                collection.insert_one(output)
            except Exception as e:
                print(f"Error writing result to MongoDB: {str(e)}")

        return output

    def _load_dataset(self, dataset_path: str) -> Optional[pd.DataFrame]:
        # Fallback to loading from CSV
        try:
            return pd.read_csv(dataset_path)
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {dataset_path}")
        except pd.errors.EmptyDataError:
            print(f"Error: The dataset file at {dataset_path} is empty")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
        return None


def main():
    load_dotenv(override=True)
    # proxy_url = os.getenv("HTTP_PROXY_URL")
    # proxy = httpx.Client(proxies=proxy_url) if proxy_url else None

    try:
        llm = ChatOpenAI(
            base_url=os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            # http_client=proxy,
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    except ValueError as e:
        print(f"Error initializing ChatOpenAI: {str(e)}")
        return

    config = {
        "additional_config_key": "value"  # Replace with actual configuration if needed
    }
    mongo_uri = os.getenv("MONGO_URI")
    data_utils = DataUtils(config=config, llm=llm, mongo_uri=mongo_uri)

    dataset_path = "./house_prices.csv"
    problem_description = """
    Predict house prices based on various features.
    The evaluation metric is Root Mean Squared Error (RMSE).
    The dataset contains information about house features and their corresponding sale prices.
    """

    state = KaggleProblemState(
        index=-1,
        problem_description=problem_description,
        dataset_path=dataset_path,
        evaluation_metric="r2_score",
        file_env_var="MY_FILE",
    )

    try:
        result = data_utils(state)
        print("Dataset Information:")
        print(result["dataset_info"])
    except Exception as e:
        print(f"An error occurred while running DataUtils: {e}")


if __name__ == "__main__":
    main()
