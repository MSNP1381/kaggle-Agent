from typing import List
from langchain.pydantic_v1 import BaseModel, Field


class EnhancedTask(BaseModel):
    task: str = Field(
        description="The original task description", default="No Task provided"
    )
    enhanced_description: str = Field(
        description="An enhanced and more detailed description of the task",
        default="no enhancement",
    )
    requires_code_output: bool = Field(
        description="Whether this task requires code execution and output",
        default=False,
    )

    requirements: List[str] = Field(
        description=(
            "outputs that are requiered for this task to as description examples are: \n"
            "'Output: model accuracy' \n"
            "'Output: feature importances'\n"
            "'Output: data frame description or info'\n"
            "'Output: report model selection on data'\n"
        )
    )
    expected_output_type: str = Field(
        description="The expected type of output if code execution is required (e.g., 'dataframe', 'plot', 'metric', "
        "'model')",
        default="No Expected Output",
    )
    dependencies: list = Field(
        description="List of tasks that this task depends on", default_factory=list
    )
    estimated_time: str = Field(
        description="Estimated time to complete this task", default="Unknown"
    )

    def __str__(self):
        return self.task
