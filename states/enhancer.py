from typing import List
from pydantic import BaseModel, Field
import yaml

"""
TODO 
- [x]: change pop to increase index
- [x]: make a list of enhanced task
- [ ]: change enhancer prompt
"""


class EnhancedTask(BaseModel):
    task: str = Field(
        description="The original task description", default="No Task provided"
    )
    summaries: str = Field(
        description="This is summary for previous tasks and quotes that you have understanded and you are writing the understanding of those.",
        default="No tasks Yet",
    )
    final_nswer: str = Field(description="Final answer to task enhancement job")

    toughts: str = Field(
        description="Toughts and rationale behind the task enhancement and what was the process from previous tasks and codes so far",
        default="No Toughts provided",
    )
    actions: List[str] = Field(
        description="Specific actions to be taken as part of this task",
        default_factory=list,
    )

    def __repr__(self) -> str:
        return self.model_dump_json(indent=1)

    def __str__(self):
        return f"""
**the Task:** {self.task} 
**summarized privious codes results:**
{self.summaries} 
**Actions for this task are:**
    -{self.final_nswer}
"""
