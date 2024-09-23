from pydantic import BaseModel, Field


class Code(BaseModel):
    imports: str = Field(description="Imports required for the solution.")
    code: str = Field(description="Code for the solution")
    description: str = Field(description="Description for the solution.")

    def __str__(self):
        return f"{self.imports}\n{self.code}"

    def __repr__(self) -> str:
        return f"{self.imports}\n--------code--------\n{self.code}\n--------desc--------\n{self.description}\n"
