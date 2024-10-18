from pydantic import BaseModel, Field

"""
TODO 
- [x]: change pop to increase index
- [x]: make a list of enhanced task
- [ ]: change enhancer prompt
"""


class EnhancedTask(BaseModel):
    final_answer: str = Field(description="Final answer to task enhancement job")

    def __repr__(self) -> str:
        return self.model_dump_json(indent=1)

    def __str__(self):
        return "\n".join(
            [
                "**the Task:**",
                f"{self.final_answer}",
            ]
        )
