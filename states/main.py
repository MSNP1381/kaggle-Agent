# from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from langchain.pydantic_v1 import BaseModel, Field
from operator import add, concat
from typing_extensions import Annotated
import yaml
from .enhancer import EnhancedTask

# from typing_extensions import TypedDict

from utils import dict_concat


class Code(BaseModel):
    """
    Represents a code solution generated by the AI assistant.

    This model encapsulates the three main components of a code solution:
    imports, main code, and description. It is designed to ensure that all
    necessary parts of a complete code solution are present and well-structured.

    Attributes:
        imports (str): A string containing all necessary import statements for the solution.
                       This should include all libraries and modules required to run the main code.

        code (str): The main body of the code solution. This should be executable Python code
                    that implements the requested functionality. Default is a placeholder comment.

        description (str): A brief explanation of what the code does, how it works, and any
                           important considerations or assumptions made in the implementation.

    Note:
        All fields are required to be non-empty when creating a Code instance.
        The 'code' field has a default value to ensure it's always present, even if empty.
    """

    imports: str = Field(
        description="Imports required for the solution.",# default="# no imports"
    )
    code: str = Field(
        description="Code for the solution", #default="#no Code for this task"
    )
    description: str = Field(description="Description for the solution.")

    def __str__(self):
        return f"{self.imports}\n{self.code}"
    def __repr__(self) -> str:
        return f"{self.imports}\n--------code--------\n{self.code}\n--------desc--------\n{self.description}\n"
        
        


class KaggleProblemState(BaseModel):
    problem_description: str = Field(default="")
    dataset_path: str = Field(default="")

    dataset_info: Optional[str] = Field(default="")
    current_task: Optional[str] = Field(default="")
    model_info: Dict[str, Any] = Field(default=None)
    # planned_tasks: List[str] = Field(default_factory=list)

    # previous_tasks: Annotated[List[str], add] = field(default_factory=list)
    task_codes_results: Annotated[List[Tuple[EnhancedTask, Code, str]], add] = Field(
        default=None
    )
    # task_results: Dict[str, Any] = field(default_factory=dict)
    # model_info: Dict[str, Any] = Field(default_factory=dict)
    planned_tasks: List[str] = Field(default=None)
    evaluation_metric: Optional[str] = Field(default=None)
    best_score: Optional[float] = Field(default=None)
    enhanced_task: EnhancedTask = Field(default=None)

    def update_task(self, task: str):
        if self.current_task:
            self.previous_tasks.append(self.current_task)
        self.current_task = task

    def update_model_info(self, info: Dict[str, Any]):
        self.model_info.update(info)

    def update_planned_tasks(self, tasks: List[str]):
        self.planned_tasks = tasks

    def update_best_score(self, score: float):
        if self.best_score is None or score > self.best_score:
            self.best_score = score

    def set_evaluation_metric(self, metric: str):
        self.evaluation_metric = metric

    def get_task_results(self):
        l = []
        for cr in self.task_codes_results:
            (enh_task, code, result) = cr
            l.append(
                f"""
            task description : 
            `
            {str(enh_task)}
            `
            ---------------------------------------
            generated code :
            ``` python
            {str(code)}
            ```
            ---------------------------------------- 
            output result :
            `
            {result}
            `
            """
            )
        return "\n".join(l)
    def __str__(self) -> str:
        return self.json()
    
    def __repr__(self) -> str:
        return self.json(indent=1)