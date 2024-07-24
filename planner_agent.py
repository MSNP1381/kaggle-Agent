import httpx
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from code_generation_agent import CodeGenerationAgent, Code
from typing import Dict, Any, Union, List

from prompts.prompts import PLANNER_PROMPT
from states.main import KaggleProblemState


class KaggleProblemPlanner:
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
            response = self.llm.invoke(self.planner_prompt.format_messages(
                problem_description=state.problem_description,
                state=str(state.__dict__)
            ), config=self.config)
            tasks = response.content.strip().split("\n")
            return tasks
        except Exception as e:
            print(f"An error occurred while generating the plan: {e}")
            return []

    def __call__(self, state: KaggleProblemState):
        initial_plan = self.plan(state)
        return {"planned_tasks": initial_plan}
# Ensure to handle specific exceptions as needed
