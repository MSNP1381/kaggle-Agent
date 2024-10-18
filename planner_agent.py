import json
import os
import logging
from injector import inject
from langchain_openai import ChatOpenAI
from typing import List
from pydantic import BaseModel, Field
from prompts.prompts import PLANNER_PROMPT
from states.main import KaggleProblemState
from langchain.output_parsers import PydanticOutputParser
from states.memory import MemoryAgent
from utils import state2doc_write

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Task(BaseModel):
    task_title: str = Field(description="Title of the task")
    task_details: str = Field(description="Details of the task")


class Plan(BaseModel):
    tasks: List[Task] = Field(description="List of tasks for a plan with their details")

    @property
    def to_list(self) -> List[str]:
        return [f"** {task.task_title} **\n{task.task_details}" for task in self.tasks]

    class Config:
        json_schema_extra = {
            "example": {
                "tasks": [
                    {
                        "task_title": "Data Collection and Understanding",
                        "task_details": "Gather the dataset, understand its structure, and identify the target variable and features.",
                    },
                    {
                        "task_title": "Data Preprocessing",
                        "task_details": "Clean the dataset, handle missing values, encode categorical variables, and normalize/standardize numerical features.",
                    },
                    {
                        "task_title": "Exploratory Data Analysis (EDA)",
                        "task_details": "Perform statistical analysis, create visualizations (histograms, scatter plots, correlation matrices), and identify patterns and outliers.",
                    },
                    {
                        "task_title": "Feature Engineering",
                        "task_details": "Create new features, perform feature selection, and apply dimensionality reduction techniques if necessary.",
                    },
                    {
                        "task_title": "Model Selection",
                        "task_details": "Research and choose appropriate machine learning algorithms based on the problem type (classification, regression, etc.) and data characteristics.",
                    },
                    {
                        "task_title": "Model Training and Validation",
                        "task_details": "Split the data into training and validation sets, train multiple models, and perform cross-validation to assess their performance.",
                    },
                    {
                        "task_title": "Hyperparameter Tuning",
                        "task_details": "Use techniques like grid search, random search, or Bayesian optimization to fine-tune the model hyperparameters.",
                    },
                    {
                        "task_title": "Model Evaluation",
                        "task_details": "Evaluate the best-performing model on a held-out test set using appropriate metrics (e.g., accuracy, F1-score, RMSE).",
                    },
                    {
                        "task_title": "Model Interpretation",
                        "task_details": "Analyze feature importance, use techniques like SHAP values to interpret the model's decisions, and generate insights.",
                    },
                    {
                        "task_title": "Documentation and Reporting",
                        "task_details": "Create a comprehensive report detailing the problem, methodology, results, and insights. Prepare visualizations and explanations for stakeholders.",
                    },
                ]
            }
        }


class KaggleProblemPlanner:
    """
    A class to plan and manage tasks for solving a Kaggle machine learning problem.
    """

    @inject
    def __init__(self, config, llm: ChatOpenAI, memory: MemoryAgent):
        """
        Initializes the KaggleProblemPlanner with configuration, proxy, and notebook executor.

        Args:
            config (dict): Configuration settings for the planner.
            llm (ChatOpenAI): Language model for generating plans.
            memory (MemoryAgent): Memory agent for managing problem-solving state.
        """
        self.config = config
        self.llm = ChatOpenAI(
            base_url="https://api.avalapis.ir/v1",
            model="gpt-4o",
            temperature=0.75,
        )
        self.memory = memory
        self.planner_prompt = PLANNER_PROMPT

    def plan(self, state: KaggleProblemState) -> List[str]:
        """
        Creates or updates a plan to solve the Kaggle problem based on the current state.

        Args:
            state (KaggleProblemState): The current state of the problem-solving process.

        Returns:
            List[str]: A list of planned tasks.
        """
        try:
            logger.info("Starting to generate plan")
            parser = PydanticOutputParser(pydantic_object=Plan)

            logger.debug(
                f"Invoking LLM with problem description: {state.problem_description[:100]}..."
            )
            response: Plan = (self.planner_prompt | self.llm | parser).invoke(
                {
                    "problem_description": state.problem_description,
                    "format_instructions": parser.get_format_instructions(),
                    "quantitative_analysis": state.quantitative_analysis,
                    "qualitative_analysis": state.qualitative_analysis,
                },
                config=self.config,
            )

            logger.info("Plan generated successfully")
            logger.info(f"Generated plan tasks count: {len(response.tasks)}")

            tasks = response.to_list
            return tasks

        except Exception as e:
            logger.error(
                f"An error occurred while generating the plan: {e}", exc_info=True
            )
            return []

    def load_plan_from_json(self, file_path: str = "./plan/plan.json") -> List[str]:
        """
        Loads a plan from a JSON file.

        Args:
            file_path (str): Path to the JSON file containing the plan.

        Returns:
            List[str]: A list of planned tasks.
        """
        try:
            with open(file_path, "r") as f:
                plan_data = json.load(f)
                return plan_data
        except Exception as e:
            logger.error(f"Error loading plan from JSON: {e}", exc_info=True)
            return []

    def save_plan_to_json(self, plan: List[str], file_path: str = "./plan/plan.json"):
        """
        Saves the current plan to a JSON file.

        Args:
            plan (List[str]): The current plan.
            file_path (str): Path to save the JSON file.
        """
        try:

            with open(file_path, "w") as f:
                json.dump(plan, f, indent=2)
            logger.info(f"Plan saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving plan to JSON: {e}", exc_info=True)

    def __call__(
        self,
        state: KaggleProblemState,
        gen_plan: bool = True,
        plan_file: str = "plan.json",
    ):
        logger.info("KaggleProblemPlanner called")
        logger.debug("Writing state to document")
        state2doc_write(state)

        logger.info("Initializing document retrieval")
        self.memory.init_doc_retrieve()

        if os.getenv("no_plan"):
            initial_plan = self.load_plan_from_json()
        else:
            initial_plan = self.plan(state)
            self.save_plan_to_json(initial_plan, "./plan/plan.json")

        return {"planned_tasks": initial_plan}


# If you want to test the planner
if __name__ == "__main__":
    # Set up a basic configuration and test state
    test_config = {}
    test_llm = ChatOpenAI()  # You might need to configure this properly
    test_memory = MemoryAgent(
        llm=test_llm, mongo=None
    )  # You might need to configure this properly
    test_state = KaggleProblemState(
        problem_description="Test problem description",
        quantitative_analysis="Test quantitative analysis",
        qualitative_analysis="Test qualitative analysis",
    )

    planner = KaggleProblemPlanner(config=test_config, llm=test_llm, memory=test_memory)
    result = planner(test_state)
    print(result)
