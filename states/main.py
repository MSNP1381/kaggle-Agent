from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from langchain.pydantic_v1 import BaseModel, Field
from operator import add, concat
from typing_extensions import Annotated
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

    Example:
        ```python
        solution = Code(
            imports="import numpy as np\nfrom sklearn.model_selection import train_test_split",
            code="def preprocess_data(X, y):\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n    return X_train, X_test, y_train, y_test",
            description="This function preprocesses the input data by splitting it into training and testing sets using an 80-20 split ratio."
        )
        ```

    Note:
        All fields are required to be non-empty when creating a Code instance.
        The 'code' field has a default value to ensure it's always present, even if empty.
    """

    imports: str = Field(description="Imports required for the solution.")
    code: str = Field(description="Code for the solution", default="#no Code for this task")
    description: str = Field(description="Description for the solution.")

    def __str__(self):
        return f"{self.imports}\n{self.code}"


@dataclass
class KaggleProblemState:
    problem_description: str
    dataset_path: str = ""

    dataset_info: str = field(default_factory=str)
    current_task: str = ""
    previous_tasks: Annotated[List[str], add] = field(default_factory=list)
    task_codes_results: Annotated[Dict[str, Tuple[Code, str]], dict_concat] = field(default_factory=dict)
    # task_results: Dict[str, Any] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)
    planned_tasks: List[str] = field(default_factory=list)
    evaluation_metric: Optional[str] = None
    best_score: Optional[float] = None
    enhanced_task: EnhancedTask = None

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
        l=[]
        for task,cr in self.task_codes_results.items():
            (code,result)=cr
            l.append(f"""
            task description : 
            `
            {task}
            `
            ---------------------------------------
            generated code :
            ``` python
            {code}
            ```
            ---------------------------------------- 
            output result :
            `
            {result}
            `
            """)
        return "\n\n".join(l)