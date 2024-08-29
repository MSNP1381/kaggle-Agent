from typing import Any, Dict, List, Optional, Tuple
from langchain.pydantic_v1 import BaseModel, Field
from operator import add
from typing_extensions import Annotated

from states.code import Code
from .enhancer import EnhancedTask
from utils import dict_concat


class KaggleProblemState(BaseModel):
    index: int = Field(default=-1)
    problem_description: str = Field(default="")
    dataset_path: str = Field(default="")
    challenge_url: str = Field(default="")
    dataset_info: Optional[str] = Field(default="")
    current_task: Optional[str] = Field(default="")
    model_info: Dict[str, Any] = Field(default=None)
    task_codes_results: List[Tuple[EnhancedTask, Code, str]] = Field(default=[])
    planned_tasks: List[str] = Field(default=None)
    evaluation_metric: Optional[str] = Field(default=None)
    best_score: Optional[float] = Field(default=None)
    enhanced_tasks: Annotated[List[EnhancedTask], add] = Field(default=None)
    file_env_var:str=None
    def get_task_results(self, to_str=True):
        l = []
        for index_, cr in enumerate(self.task_codes_results):
            (enh_task, code, result) = cr
            l.append(
                f"""
Executed Task No.{index_+1}

**Task Description :** 
`
{str(enh_task)}
`
----------------------------------------
**Generated Code :**
``` python
{str(code)}
```
---------------------------------------- 
**Output Result :**
`
{result}
`
----------------------------------------
            """
            )
        return "\n".join(l) if to_str else l

    def get_future_tasks(self):
        i = self.index + 1
        tasks = self.planned_tasks[i:]
        return "".join([f"- {i}\n" for i in tasks])

    def get_planned_tasks(self) -> str:
        index = self.index + 1
        tasks = self.planned_tasks[:index]
        return "\n-" + "".join([f"- {i}\n" for i in tasks])

    def __str__(self) -> str:
        return self.json()

    def __repr__(self) -> str:
        return self.json(indent=1)
