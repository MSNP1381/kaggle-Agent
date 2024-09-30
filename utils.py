import json

from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from typing import List, Optional, Union
import json

from e2b_code_interpreter.models import Error, Result, MIMEType


@dataclass
class CellOutput:
    output_type: str
    name: str
    text: str

    def to_json(self) -> str:
        # Convert the dataclass to a dictionary and then to a JSON string
        return json.dumps(asdict(self))


class CellError(Exception):
    def __init__(self, ename, evalue, traceback):
        self.ename = ename
        self.evalue = evalue
        self.traceback = traceback

    def __repr__(self) -> str:
        return f"error: {self.ename}"

    # @property
    # def ename(self)->str:
    #     return self.name
    # @property
    # def evalue(self)->str:
    #     return self.value
    # @property
    # def traceback(self)->str:
    #     return self.traceback_


class CellResult:
    def __init__(self, result):
        # super().__init__(result_instance.is_main_result,result_instance.extra)
        self.result = result

    @property
    def output_str(self) -> str:
        self.result

    def __str__(self) -> str:
        return self.result

    # def __repr__(self) -> str:
    #     return self.result


class NotebookExecutorInterface(ABC):

    def create_nb(self) -> str:
        pass

    def __init__(self, execution_instance) -> None:
        self.executor = None
        self.is_restarted = False

    def test_and_execute(self, new_code: str) -> Union[CellResult, List[CellResult]]:
        pass

    def reset(self) -> None:
        pass


def dict_concat(a, b):
    return {**a, **b}


def cc(s: str):
    return s.replace("\\n", "\n")


def exec2s(data: Union[CellResult, List[CellResult]]) -> str:
    out_s = ""
    if isinstance(data, List):
        out_s = "\n---\n".join(map(str, data))
    else:
        out_s = str(data)
    return out_s

def get_top_10_percent_mean(json_path: str) -> float:
    """
    Calculate the mean of the top 10 percent score values from the provided JSON schema.

    :param json_path: A string containing the path to the JSON file with score values.
    :return: The mean of the top 10 percent score values.
    """

    with open(json_path, 'r') as file:
        json_data = json.load(file)

    submissions = json_data.get("submissions", [])
    if not submissions:
        return 0.0

    scores = []
    for submission in submissions:
        score_str = submission.get("score")
        if score_str is not None:
            try:
                score = float(score_str)
                scores.append(score)
            except ValueError:
                continue

    if not scores:
        return 0.0

    scores.sort(reverse=True)
    top_10_percent_count = max(1, len(scores) // 10)
    top_10_percent_scores = scores[:top_10_percent_count]

    return sum(top_10_percent_scores) / len(top_10_percent_scores)

def append_url(base_url: str, sub_url: str, use_https: bool = True) -> str:
    """
    Append a sub URL to a base URL and add the appropriate protocol (http or https).

    :param base_url: The base URL to which the sub URL will be appended.
    :param sub_url: The sub URL to append to the base URL.
    :param use_https: Boolean flag to determine whether to use https (default) or http.
    :return: The complete URL with the appropriate protocol.
    """
    protocol = "https://" if use_https else "http://"
    if not base_url.startswith("http://") and not base_url.startswith("https://"):
        base_url = protocol + base_url
    if not base_url.endswith("/"):
        base_url += "/"
    return base_url + sub_url
