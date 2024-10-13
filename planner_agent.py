from pprint import pprint
import httpx
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import Dict, Any, Union, List
from pydantic import BaseModel, Field

from prompts.prompts import PLANNER_PROMPT
from states.main import KaggleProblemState
from langchain.output_parsers import PydanticOutputParser


class Plan(BaseModel):

    description: str = Field(
        description="detailed description about dataset info and problem description and give reasoning about problem and procedure"
    )
    tasks: List[str] = Field(
        description="A list of tasks to be performed to solve the ML problem",
    )


class KaggleProblemPlanner:
    """
    A class to plan and manage tasks for solving a Kaggle machine learning problem.
    """

    def __init__(self, config, proxy, llm: ChatOpenAI):
        """
        Initializes the KaggleProblemPlanner with configuration, proxy, and notebook executor.

        Args:
            config (dict): Configuration settings for the planner.
            proxy (httpx.Client): HTTP client for proxy settings.
            nb_executor (Any): Notebook executor for executing generated code.
        """
        self.config = config
        # self.nb_executor = nb_executor
        self.llm = llm
        # self.code_generation_agent = CodeGenerationAgent(config=config, proxy=proxy, nb_executor=self.nb_executor)
        self.planner_prompt = PLANNER_PROMPT

    def plan(self, state: KaggleProblemState) -> List[str]:
        """
        Creates or updates a plan to solve the Kaggle problem based on the current state.

        Args:
            state (Any): The current state of the problem-solving process, including problem description and previous results.

        Returns:
            List[str]: A list of planned tasks .
        """

        try:
            parser = PydanticOutputParser(pydantic_object=Plan)
            response: Plan = (self.planner_prompt | self.llm | parser).invoke(
                {
                    "problem_description": state.problem_description,
                    "dataset_info": str(state.dataset_info),
                    "quantitative_analysis": state.quantitative_analysis,
                    "qualitative_analysis": state.qualitative_analysis,
                    "format_instructions": parser.get_format_instructions(),
                },
                config=self.config,
            )
            # pprint(response.description)
            tasks = response.tasks
            return tasks
        except Exception as e:
            print(f"An error occurred while generating the plan: {e}")
            return []

    def __call__(self, state: KaggleProblemState):
        initial_plan = self.plan(state)

        return {"planned_tasks": initial_plan}


# Ensure to handle specific exceptions as needed
