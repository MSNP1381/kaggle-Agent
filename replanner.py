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
    def __init__(self, config, proxy, llm: ChatOpenAI):
        self.config = config
        self.llm = llm
        self.output_parser = PydanticOutputParser(pydantic_object=RePlanDecision)

        self.re_plan_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an AI assistant specialized in re-planning machine learning projects for Kaggle competitions. Your objective is to ensure that the project plan remains effective and aligned with the competition goals based on the latest execution results.

**Objectives:**
1. **Alignment:** Ensure the current plan aligns with the overall problem objectives.
2. **Efficiency:** Identify and address any inefficiencies or bottlenecks in the current workflow.
3. **Adaptability:** Incorporate new insights or resolve issues highlighted by recent task executions.

**Process:**
- **Analyze** the problem description and current project state.
- **Review** the last executed task and its outcomes.
- **Assess** the effectiveness of the existing plan.
- **Decide** whether adjustments are necessary.
- **Propose** a revised set of tasks if modifications are needed.

{format_instructions}

**Your response should include:**
1. **Decision:** Whether the plan needs to be changed.
2. **Reasoning:** Explanation for your decision.
3. **New Plan:** A list of revised tasks (if adjustments are required)."""
                ),
                (
                    "human",
                    """**Problem Description:**
{problem_description}

**Current State:**
{state}

**Last Executed Task:**
{last_task}

**Execution Result:**
{execution_result}

**Current Plan:**
{current_plan}

**Question:**
Based on the above information, should the project plan be adjusted? If so, please provide a revised list of tasks."""
                ),
            ]
        )

    def re_plan(self, state: KaggleProblemState) -> List[str]:
        last_task = (
            state.enhanced_tasks[state.index].enhanced_description
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
            "modelInfo": state.modelInfo,
            "evaluation_metric": state.evaluation_metric,
            "best_score": state.best_score,
        }
        return str(formatted_state)
