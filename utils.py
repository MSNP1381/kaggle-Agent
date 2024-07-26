import json
import subprocess
from typing import Any, Dict, List, Union

# from states.main import KaggleProblemState
from prompts.utils import DATASET_ANALYSIS_PROMPT

# from states.main import Code

# from task_enhancer import EnhancedTask


class VenvExecutionError(Exception):
    """Custom exception for virtual environment execution errors."""

    pass


def exec_in_venv(code, venv_path="./.venv"):
    # Construct the Python command
    python_executable = f"{venv_path}/bin/python"

    # Use subprocess to run the code in the specified virtual environment
    process = subprocess.Popen(
        [python_executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Capture output and errors
    stdout, stderr = process.communicate()

    # Check if the process executed successfully
    if process.returncode != 0:
        error_message = (
            f"Execution failed with return code {process.returncode}.\nStderr: {stderr}"
        )
        raise VenvExecutionError(error_message)

    return stdout


def _get_relevant_previous_code(self, state: "KaggleProblemState") -> str:
    relevant_code = []
    for task, code in state.task_codes_results.items():
        if self._is_relevant(state.current_task, task):
            relevant_code.append(
                f"# Code for task: {self._get_task_description(task)}\n{code}\n"
            )
    return "\n".join(relevant_code)


def _get_relevant_previous_results(self, state: "KaggleProblemState") -> str:
    relevant_results = []
    for task, result in state.task_results.items():
        if self._is_relevant(state.current_task, task):
            relevant_results.append(
                f"# Result for task: {self._get_task_description(task)}\n{result}\n"
            )
    return "\n".join(relevant_results)


def _is_relevant(
    self,
    current_task: Union[str, "EnhancedTask"],
    previous_task: Union[str, "EnhancedTask"],
) -> bool:
    current_task_desc = self._get_task_description(current_task)
    previous_task_desc = self._get_task_description(previous_task)

    current_keywords = set(current_task_desc.lower().split())
    previous_keywords = set(previous_task_desc.lower().split())
    return len(current_keywords.intersection(previous_keywords)) > 0


def _format_project_state(self, state: "KaggleProblemState") -> str:
    formatted_state = {
        "dataset_info": state.dataset_info,
        "previous_tasks": [
            self._get_task_description(task) for task in state.previous_tasks
        ],
        "model_info": state.model_info,
        "evaluation_metric": state.evaluation_metric,
        "best_score": state.best_score,
    }
    return str(formatted_state)


def _extract_code(self, response: str) -> List[str]:
    # Simple extraction of code blocks
    code_blocks = []
    in_code_block = False
    for line in response.split("\n"):
        if line.strip().startswith("```python"):
            in_code_block = True
            continue
        elif line.strip() == "```" and in_code_block:
            in_code_block = False
            continue
        if in_code_block:
            code_blocks.append(line)
    return code_blocks


def _is_code_complete(self, code: str, task: Union[str, "EnhancedTask"]) -> bool:
    # This is a simplified check. You might want to implement a more sophisticated one.
    task_desc = self._get_task_description(task).lower()
    if "model" in task_desc and "fit" in code:
        return True
    if "preprocess" in task_desc and "transform" in code:
        return True
    if "evaluate" in task_desc and "score" in code:
        return True
    # Default to False if we're not sure
    return False


def dict_concat(a, b):
    return {**a, **b}
