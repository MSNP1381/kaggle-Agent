import httpx
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

class ReplanDecision(BaseModel):
    """
    Represents the decision made by the replanner.

    Attributes:
        changes_needed (bool): Indicates whether changes to the plan are needed.
        new_plan (Optional[List[str]]): The new plan if changes are needed, otherwise None.
        reasoning (str): The reasoning behind the decision.
    """
    changes_needed: bool = Field(description="Whether changes to the plan are needed")
    new_plan: Optional[List[str]] = Field(description="The new plan if changes are needed", default=None)
    reasoning: str = Field(description="The reasoning behind the decision")

class KaggleProblemReplanner:
    def __init__(self, config, proxy):
        self.config = config
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", http_client=proxy, temperature=0)
        self.output_parser = PydanticOutputParser(pydantic_object=ReplanDecision)
        
        self.replan_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant tasked with replanning a Kaggle machine learning project based on the latest execution results.
            Given the current state of the project and the output of the last executed task, determine if the plan needs to be adjusted.

            Analyze the execution result carefully and determine if it requires changes to the plan.
            If changes are needed, provide a new list of planned tasks.
            If no changes are needed, keep the current plan.

            {format_instructions}

            Ensure your response includes a clear decision on whether changes are needed, the reasoning behind your decision, and a new plan if applicable."""),
            ("human", """Problem Description:
            {problem_description}

            Current State:
            {state}

            Last executed task: {last_task}
            Execution result: {execution_result}

            Current plan:
            {current_plan}

            Based on this information, should the plan be changed? If so, what should the new plan be?""")
        ])

    def replan(self, state):
        last_task = state.current_task
        execution_result = state.task_results.get(last_task, "No result available")
        current_plan = "\n".join(state.planned_tasks)
        
        response = self.llm.invoke(
            self.replan_prompt.format_messages(
                problem_description=state.problem_description,
                state=str(state.__dict__),
                last_task=last_task,
                execution_result=execution_result,
                current_plan=current_plan,
                format_instructions=self.output_parser.get_format_instructions()
            ),
            config=self.config
        )

        try:
            replan_decision = self.output_parser.parse(response.content)
            if replan_decision.changes_needed:
                return replan_decision.new_plan
            else:
                return state.planned_tasks
        except Exception as e:
            print(f"Error parsing replanner output: {e}")
            return state.planned_tasks  # Fall back to current plan if parsing fails

if __name__ == "__main__":
    # Example usage
    class MockState:
        def __init__(self):
            self.problem_description = "Predict house prices"
            self.current_task = "Perform exploratory data analysis"
            self.task_results = {"Perform exploratory data analysis": "Found high correlation between house size and price"}
            self.planned_tasks = ["Perform exploratory data analysis", "Feature engineering", "Model selection", "Model training", "Model evaluation"]

    replanner = KaggleProblemReplanner(config={}, proxy=None)  # Provide appropriate config and proxy
    mock_state = MockState()
    new_plan = replanner.replan(mock_state)
    print("New plan:", new_plan)