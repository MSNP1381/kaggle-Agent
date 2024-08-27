import json
import subprocess
from typing import Any, Dict, List, Literal, Union
import yaml
import json

# from states.main import KaggleProblemState

from prompts.utils import DATASET_ANALYSIS_PROMPT

from dataclasses import dataclass, asdict


@dataclass
class CellOutput:
    output_type: str
    name: str
    text: str

    def to_json(self) -> str:
        # Convert the dataclass to a dictionary and then to a JSON string
        return json.dumps(asdict(self))


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
        raise (error_message)

    return stdout


def dict_concat(a, b):
    return {**a, **b}


def cc(s: str):
    return s.replace("\\n", "\n")
