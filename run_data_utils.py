import os
import json
import httpx
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from data_utils import DataUtils
from states.main import KaggleProblemState

def load_dataset(dataset_path):
    try:
        return pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The dataset file at {dataset_path} is empty")
        return None
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Initialize proxy if needed
    proxy_url = os.getenv("HTTP_PROXY_URL")
    proxy = httpx.Client(proxies=proxy_url) if proxy_url else None

    # Initialize OpenAI LLM
    try:
        llm = ChatOpenAI(
            base_url=os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            http_client=proxy,
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    except ValueError as e:
        print(f"Error initializing ChatOpenAI: {str(e)}")
        return

    # Initialize DataUtils with configuration
    config = {
        "additional_config_key": "value"  # Replace with actual configuration if needed
    }
    data_utils = DataUtils(config=config, proxy=proxy, llm=llm)

    # Define the dataset path and problem description
    dataset_path = "./house_prices.csv"  # Ensure this path is correct
    problem_description = """
    Predict house prices based on various features.
    The evaluation metric is Root Mean Squared Error (RMSE).
    The dataset contains information about house features and their corresponding sale prices.
    Remember, if you want to get the dataset, the dataset path is stored in the environment variable: MY_FILE
    """

    # Load the dataset
    dataset = load_dataset(dataset_path)
    if dataset is None:
        return

    # Create KaggleProblemState instance
    state = KaggleProblemState(
        index=-1,
        problem_description=problem_description,
        dataset_path=dataset_path,
        evaluation_metric="r2_score",
    )

    # Invoke DataUtils with the state
    try:
        result = data_utils(state)
        print("Dataset Information:")
        print(json.dumps(json.loads(result["dataset_info"]), indent=4))
    except Exception as e:
        print(f"An error occurred while running DataUtils: {e}")
    finally:
        if proxy:
            proxy.close()

if __name__ == "__main__":
    main()