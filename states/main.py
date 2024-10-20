from operator import add
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from states.code import Code

from .enhancer import EnhancedTask


class KaggleProblemState(BaseModel):
    index: int = Field(default=-1)
    quantitative_analysis: str = Field(default="")
    qualitative_analysis: str = Field(default="")
    problem_description: str = Field(default="")
    dataset_path: str = Field(default="./ongoging/train.csv")
    test_dataset_path: str = Field(default="./ongoging/test.csv")
    challenge_url: str = Field(default="")
    dataset_info: Optional[str] = Field(default="")
    current_task: Optional[str] = Field(default="")
    modelInfo: Dict[str, Any] = Field(default=None)
    task_codes_results: List[Tuple[EnhancedTask, Code, str]] = Field(default=[])
    planned_tasks: List[str] = Field(default=None)
    evaluation_metric: Optional[str] = Field(default=None)
    best_score: Optional[float] = Field(default=None)
    enhanced_tasks: Annotated[List[EnhancedTask], add] = Field(default=[])
    file_env_var: str = None

    def get_executed_codes(self, last_task_num=-1) -> str:
        code_list = []
        if len(self.task_codes_results) == 0:
            return ""
        if last_task_num == -1:
            last_task_num = len(self.task_codes_results)

        for _, code, _ in self.task_codes_results[:-last_task_num:-1]:
            code_list.append("\n# %% \n")
            code_list.append(str(code))

        return "".join(code_list)

    def get_task_results(self, last_task_num=0, to_str=True):
        task_list = ["** Number of executed task results : {last_task_num}\n"]
        for index_, cr in enumerate(self.task_codes_results[:-last_task_num:-1]):
            (enh_task, code, result) = cr
            task_list.append(
                f"""
Executed Task No.{index_+1}

**Task Description :**
`
{str(enh_task.task)}
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
----------------------------------------"""
            )
        return "\n".join(task_list) if to_str else task_list

    def get_future_tasks(self):
        i = self.index + 1
        tasks = self.planned_tasks[i:]
        return "".join([f"- {i}\n" for i in tasks])

    def get_planned_tasks(self) -> str:
        index = self.index + 1
        tasks = self.planned_tasks[:index]
        output_str = "** Number of executed tasks: ** \n "
        for i in tasks:
            output_str += f"- {i}\n"
        return output_str

    def __str__(self) -> str:
        return self.model_dump_json()

    def __repr__(self) -> str:
        return self.model_dump_json(indent=1)

    def get_previous_result(self, last_n: int = 1) -> str:
        """
        Get the result of the previous task(s).

        Args:
            last_n (int): Number of previous results to retrieve. Defaults to 1.

        Returns:
            str: A formatted string containing the previous task result(s).
        """
        if not self.task_codes_results:
            return "No previous results available."

        results = []
        for i, (enhanced_task, _, result) in enumerate(
            self.task_codes_results[-last_n:][::-1], 1
        ):
            results.append(
                f"Previous Task {i}:\n"
                f"Task: {enhanced_task.final_answer}\n"
                f"Result:\n{result}\n"
            )

        return "\n".join(results)
