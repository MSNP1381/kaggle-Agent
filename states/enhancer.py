from typing import List
from langchain.pydantic_v1 import BaseModel, Field
import yaml
"""
TODO 
- [ ]: change pop to increase index
- [ ]: make a list of enhanced task
- [ ]: change enhancer prompt
"""

class EnhancedTask(BaseModel):
    task: str = Field(
        description="The original task description", default="No Task provided"
    )
    enhanced_description: str = Field(
        description="An enhanced and more detailed description of the task, including reasoning and rationale",
        default="No enhancement",
    )
    requires_code_output: bool = Field(
        description="Whether this task requires code execution and output",
        default=False,
    )
    requirements: List[str] = Field(
        description=(
            "Examples of required outputs for this task:\n"
            "- 'Output: model accuracy'\n"
            "- 'Output: feature importances'\n"
            "- 'Output: data frame description or info'\n"
            "- 'Output: report model selection on data'\n"
        )
    )
    expected_output_type: str = Field(
        description="The expected type of output if code execution is required (e.g., 'dataframe', 'plot', 'metric', 'model')",
        default="No Expected Output",
    )
    dependencies: List[str] = Field(
        description="List of tasks that this task depends on", default_factory=list
    )

    reasoning: str = Field(
        description="Reasoning and rationale behind the task enhancement",
        default="No reasoning provided",
    )
    actions: List[str] = Field(
        description="Specific actions to be taken as part of this task",
        default_factory=list,
    )
    def __repr__(self) -> str:
        return self.json(indent=1)
    def __str__(self):
        return f"Task: {self.task}\nEnhanced Description: {self.enhanced_description}\nReasoning: {self.reasoning}\nActions: {', '.join(self.actions)}"
