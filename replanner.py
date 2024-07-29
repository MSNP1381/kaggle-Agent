import httpx
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from states.main import KaggleProblemState


class RePlanDecision(BaseModel):
    """
    Represents the decision made by the replanner.

    Attributes:
        changes_needed (bool): Indicates whether changes to the plan are needed.
        new_plan (Optional[List[str]]): The new plan if changes are needed, otherwise None.
        reasoning (str): The reasoning behind the decision.
    """

    changes_needed: bool = Field(description="Whether changes to the plan are needed")
    new_plan: Optional[List[str]] = Field(
        description="The new plan if changes are needed", default=None
    )
    reasoning: str = Field(description="The reasoning behind the decision")


class KaggleProblemRePlanner:
    def __init__(self, config, proxy):
        self.config = config
        self.llm = ChatOpenAI(model="gpt-4o-mini", http_client=proxy, temperature=0)
        self.output_parser = PydanticOutputParser(pydantic_object=RePlanDecision)

        self.re_plan_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an AI assistant tasked with re planning a Kaggle machine learning project based on 
            the latest execution results. Given the current state of the project and the output of the last executed 
            task, determine if the plan needs to be adjusted.

            Analyze the execution result carefully and determine if it requires changes to the plan.
            If changes are needed, provide a new list of planned tasks.
            If no changes are needed, keep the current plan.

            {format_instructions}

            Ensure your response includes a clear decision on whether changes are needed, the reasoning behind your 
            decision, and a new plan if applicable.""",
                ),
                (
                    "human",
                    """Problem Description:
            {problem_description}

            Current State:
            {state}

            Last executed task: {last_task}
            Execution result: {execution_result}

            Current plan:
            {current_plan}

            Based on this information, should the plan be changed? If so, what should the new plan be?""",
                ),
            ]
        )

    def re_plan(self, state: KaggleProblemState) -> List[str]:
        last_task = (
            state.enhanced_task.enhanced_description
            if hasattr(state.enhanced_task, "task")
            else str(state.enhanced_task)
        )
        execution_result = self._get_execution_result(state.task_results, last_task)
        current_plan = "\n".join(state.planned_tasks)

        response = self.llm.invoke(
            self.re_plan_prompt.format_messages(
                problem_description=state.problem_description,
                state=self._format_state_for_prompt(state),
                last_task=last_task,
                execution_result=execution_result,
                current_plan=current_plan,
                format_instructions=self.output_parser.get_format_instructions(),
            ),
            config=self.config,
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

    def _get_execution_result(
        self, task_results: Dict[str, Any], task_name: str
    ) -> str:
        for key, value in task_results.items():
            if isinstance(key, str) and key == task_name:
                return str(value)
            elif hasattr(key, "task") and key.task == task_name:
                return str(value)
        return "No result available"

    def _format_state_for_prompt(self, state: "KaggleProblemState") -> str:
        formatted_state = {
            "dataset_info": state.dataset_info,
            "previous_tasks": state.previous_tasks,
            "model_info": state.model_info,
            "evaluation_metric": state.evaluation_metric,
            "best_score": state.best_score,
        }
        return str(formatted_state)
