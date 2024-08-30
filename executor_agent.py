import pandas as pd
from IPython.display import display, HTML

from nbexecutor import (
    NBExecutor,
)  # Assuming NBExecutor is in a file named nbexecutor.py
from states.main import KaggleProblemState
from utils import NotebookExecutorInterface, cc, exec2s


class KaggleCodeExecutor:
    def __init__(self, nb_executor:NotebookExecutorInterface):
        self.nb_executor = nb_executor

        # Add initial imports
        initial_imports = (
            """import pandas as pd\nfrom IPython.display import display, HTML"""
        )
        # self.nb_executor.create_nb()
        self.nb_executor.test_and_execute(initial_imports)

    def execute_code(self, code, task: str):
        # Combine imports and code
        code_txt = (
            # f'\n_="""\n\n{task}\n\n"""\n\n' +
            code.imports
            + "\n"
            + code.code
        )
        output = self.nb_executor.test_and_execute(code_txt)
        return cc(exec2s(output))

    def __call__(self, state: KaggleProblemState):
        enhanced_task = state.enhanced_tasks[state.index]
        code = state.task_codes_results[-1][1]
        output = self.execute_code(code, str(state.enhanced_tasks[state.index]))
        task_codes = state.task_codes_results
        task_codes[-1] = (enhanced_task, code, str(output))
        return {"task_codes_results": task_codes}
