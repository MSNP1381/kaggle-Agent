import json

from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from typing import List, Optional, Union

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

    def process_cell_output(
        self,
    ) -> Optional[str]:
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
        out_s = "\n---\n".join(map(str,data))
    else:
        out_s = str(data)
    return out_s
