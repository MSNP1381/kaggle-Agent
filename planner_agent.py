import httpx
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import Dict, Any, Union, List
from langchain.pydantic_v1 import BaseModel, Field

from prompts.prompts import PLANNER_PROMPT
from states.main import KaggleProblemState
from langchain.output_parsers import PydanticOutputParser


class Plan(BaseModel):
    """
    A structured plan for solving a Kaggle machine learning problem.

    This model represents a series of tasks to be executed in order to solve
    a given machine learning problem. Each task is described as a string,
    providing a step-by-step guide for the problem-solving process.

    Attributes:
        tasks (List[str]): A list of task descriptions, where each task is a string
                           detailing a specific step in the problem-solving process.
    """

    tasks: List[str] = Field(
        ...,
        description="A list of tasks to be performed to solve the ML problem",
        min_items=1,
        # example=[
        #     "Task 1: Load and explore the dataset using pandas",
        #     "Task 2: Preprocess the data by handling missing values and encoding categorical variables",
        #     "Task 3: Perform feature engineering to create relevant features for house price prediction",
        # ],
    )


class KaggleProblemPlanner():
    """
    A class to plan and manage tasks for solving a Kaggle machine learning problem.
    """

    def __init__(self, config, proxy):
        """
        Initializes the KaggleProblemPlanner with configuration, proxy, and notebook executor.

        Args:
            config (dict): Configuration settings for the planner.
            proxy (httpx.Client): HTTP client for proxy settings.
            nb_executor (Any): Notebook executor for executing generated code.
        """
        self.config = config
        # self.nb_executor = nb_executor
        self.llm = ChatOpenAI(model="gpt-4o-mini", http_client=proxy, temperature=0)
        # self.code_generation_agent = CodeGenerationAgent(config=config, proxy=proxy, nb_executor=self.nb_executor)
        self.planner_prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)

    def plan(self, state: KaggleProblemState):
        """
        Creates or updates a plan to solve the Kaggle problem based on the current state.

        Args:
            state (Any): The current state of the problem-solving process, including problem description and previous results.

        Returns:
            List[str]: A list of planned tasks.
        """

        try:
            parser = PydanticOutputParser(pydantic_object=Plan)
            response = (self.planner_prompt | self.llm | parser).invoke(
                {
                    "problem_description": state.problem_description,
                    "dataset_info": str(state.dataset_info),
                    "format_instructions": parser.get_format_instructions(),
                },
                config=self.config,
            )

            tasks = response.tasks
            return tasks
        except Exception as e:
            print(f"An error occurred while generating the plan: {e}")
            return []

    def __call__(self, state: KaggleProblemState):
        initial_plan = self.plan(state)
        return {"planned_tasks": initial_plan}


# Ensure to handle specific exceptions as needed
