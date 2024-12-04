from pydantic import BaseModel, Field

#
# TODO
# - [x]: change pop to increase index
# - [x]: make a list of enhanced task
# - [ ]: change enhancer prompt
#


class EnhancedTask(BaseModel):
    original_task: str

    TASK_ANALYSIS: str = Field(description="The task analysis")
    previous_learnings: str = Field(
        description="The learnings from previous tasks and results"
    )
    code_generation_recommendation: str = Field(
        description="The code generation recommendations"
    )
    # implementation_guidance: str = Field(description="The implementation on gow to implemt code for the task and previous results")

    def __repr__(self) -> str:
        return self.model_dump_json(indent=1)

    def __str__(self):
        return "\n".join(
            [
                "**the Task:**",
                f"{self.original_task}",
                "**code genratin recommendations:**\n",
                f"{self.code_generation_recommendation}",
                "============",
                # "**implementation guidance**:\n",
                # f"{self.implementation_guidance}",
            ]
        )
