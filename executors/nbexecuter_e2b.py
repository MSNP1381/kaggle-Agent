from typing import List

from e2b_code_interpreter import CodeInterpreter

from utils import CellError, CellResult, NotebookExecutorInterface


class SandboxManager:
    def __init__(self):
        self.sandbox = None

    def __enter__(self):
        enterpretes_list = CodeInterpreter.list()
        if len(enterpretes_list) > 0:
            self.sandbox = CodeInterpreter.reconnect(enterpretes_list[0].sandbox_id)
        else:
            self.sandbox = CodeInterpreter()
            self.sandbox.keep_alive(3600)

        return self.sandbox.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.sandbox.__exit__(exc_type, exc_val, exc_tb)


class E2B_executor(NotebookExecutorInterface):
    def __init__(self, execution_instance: CodeInterpreter) -> None:
        # super().__init__(execution_instance)
        self.executor = execution_instance
        self.is_restarted = False

    def reset(self):
        self.is_restarted = True
        self.executor.notebook.restart_kernel()

    def test_and_execute(self, code) -> List[CellResult]:
        executiuon_result = self.executor.notebook.exec_cell(code)
        if executiuon_result.error:
            e = CellError(executiuon_result.error)
            raise e
        return [CellResult(i) for i in executiuon_result.results]
