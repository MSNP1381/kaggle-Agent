
import httpx
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class KaggleProblemReplanner:
    def __init__(self,config):
        self.config =config
        proxy=httpx.Client(proxy="http://127.0.0.1:2081")
        self.llm = ChatOpenAI(model="gpt-3.5-turbo",http_client=proxy)
        self.replan_prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant tasked with replanning a Kaggle machine learning project based on the latest execution results.
        Given the current state of the project and the output of the last executed task, determine if the plan needs to be adjusted.

        Problem Description:
        {problem_description}

        Current State:
        {state}

        Last executed task: {last_task}
        Execution result: {execution_result}

        Current plan:
        {current_plan}

        Your task is to:
        1. Analyze the execution result and determine if it requires changes to the plan.
        2. If changes are needed, provide a new list of planned tasks.
        3. If no changes are needed, return the current plan.

        Respond with a list of planned tasks, each on a new line.
        """)

    def replan(self, state):
        last_task = state.current_task
        execution_result = state.task_results.get(last_task, "No result available")

        response = self.llm.invoke(self.replan_prompt.format_messages(
            problem_description=state.problem_description,
            state=str(state.__dict__),
            last_task=last_task,
            execution_result=execution_result,
            current_plan="\n".join(state.planned_tasks)
        ),config=self.config)

        return response.content.strip().split("\n")
