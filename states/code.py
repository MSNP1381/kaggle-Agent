from pydantic import BaseModel, Field


class Code(BaseModel):
    code: str = Field(description="Code for the solution")

    def __str__(self):
        return f"{self.code}"

    def __repr__(self) -> str:
        return f"<code>{self.code}</code>"
