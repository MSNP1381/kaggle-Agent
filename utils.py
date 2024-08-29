import json
import json
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from typing import List, Optional,Union
import nbformat
from e2b_code_interpreter.models import Error,Result,MIMEType


@dataclass
class CellOutput:
    output_type: str
    name: str
    text: str

    def to_json(self) -> str:
        # Convert the dataclass to a dictionary and then to a JSON string
        return json.dumps(asdict(self))
    
class CellError(Exception):
    def __init__(self,err_instance:Error):
        self.err_instance=err_instance
        super().__init__(err_instance.traceback)
    @property
    def ename(self)->str:
        return self.err_instance.name
    @property
    def evalue(self)->str:
        return self.err_instance.value
    @property
    def traceback(self)->str:
        return self.err_instance.traceback
    
class CellResult(Result):
    def __init__(self, result_instance:Result):
        super().__init__(result_instance.is_main_result,result_instance.extra)
    @property
    def output_str(self)->str:
        self.__str__()
        

class NotebookExecutorInterface(ABC):
    
    
    def create_nb(self) -> str:
        pass

    
    def __init__(self,execution_instance) -> None:
        self.executor=None
        self.is_restarted=False
        

    
    def test_and_execute(self, new_code: str) ->Union[ CellResult,List[CellResult]] :
        pass

    
    def process_cell_output(self,) -> Optional[str]:
        pass
    
    def reset(self)->None:
        pass
  
    

def dict_concat(a, b):
    return {**a, **b}


def cc(s: str):
    return s.replace("\\n", "\n")
