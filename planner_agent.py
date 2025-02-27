import json
import logging
import os
from typing import List

from injector import inject
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from prompts.prompts import PLANNER_PROMPT
from states.main import KaggleProblemState
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

    def to_list(self) -> List[str]:
        task_list = []
        for task in self.tasks:
            task_list.append(
                f"""\n<TaskTitle>
{task.task_title}
</TaskTitle>
<TaskDetails>
{task.task_details}
</TaskDetails>\n
"""
            )
        return task_list


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
            model="gpt-4o",
            temperature=0.2,
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

            logger.debug(
                f"Invoking LLM with problem description: {state.problem_description[:100]}..."
            )
            response: Plan = (
                self.planner_prompt | self.llm.with_structured_output(Plan)
            ).invoke(
                {
                    "problem_description": state.problem_description,
                    "quantitative_analysis": state.quantitative_analysis,
                    "qualitative_analysis": state.qualitative_analysis,
                    "feature_recommendations": state.feature_recommendations,
                },
                config=self.config,
            )

            logger.info("Plan generated successfully")
            logger.info(f"Generated plan tasks count: {len(response.tasks)}")

            tasks = response.to_list()
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
        # self.memory.init_doc_retrieve()
        logger.info("no_plan: " + str(os.getenv("no_plan")))
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
        feature_recommendations="Test feature recommendations",
    )

    planner = KaggleProblemPlanner(config=test_config, llm=test_llm, memory=test_memory)
    result = planner(test_state)
    print(result)
