from queue import Empty
import time
from typing import IO, List
from jupyter_client import KernelManager
from utils import CellResult, NotebookExecutorInterface, CellError
from utils import CellResult, NotebookExecutorInterface, CellError
from datetime import datetime
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_notebook, new_code_cell, new_output


class JupyterExecutor(NotebookExecutorInterface):
    def __init__(self, url, token):
        # self.kc = kernel_client
        self.url = url
        self.token = token
        self.kernel()
        self.is_restarted = False
        self.notebook_path = self.create_notebook("my_notebook.ipynb", "./notebooks")

    def kernel(self):
        self.km = KernelManager(url=self.url, token=self.token)
        print("Starting kernel...")
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()

        self.kc.wait_for_ready()
        time.sleep(1)  # Give a short pause to ensure kernel is fully ready
        print("Kernel is ready.")

        # Reset the kernel by executing a command to delete all variables
        self.kc.execute("%reset -f")
        self.is_restarted = True
        print("Kernel reset.")

        # Wait until the kernel is ready
        self.kc.wait_for_ready()

    def reset(self):
        print("Resetting kernel...")
        self.kc.execute("%reset -f")
        self.is_restarted = True
        print("Kernel reset.")
        self.kc.wait_for_ready()
        time.sleep(2)  # Give a short pause to ensure kernel is fully ready
        print("Kernel is ready.")

    def actions_on_error(self, code, cell_outputs):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        error_notebook_path = os.path.join(
            os.path.dirname(self.notebook_path),
            "error_" + timestamp + "_" + os.path.basename(self.notebook_path),
        )
        os.rename(self.notebook_path, error_notebook_path)
        self.add_to_notebook(code, cell_outputs, error_notebook_path)

        self.recreate_notebook()

    def test_and_execute(
        self,
        code: str,
        notebook_name: str = "my_notebook.ipynb",
        notebook_dir: str = "./notebooks",
    ) -> List[CellResult]:
        if self.kc is None:
            self.restart_kernel()

        results = []
        errors = []

        msg_id = self.kc.execute(code)
        cell_outputs = []

        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=10)
                if msg["parent_header"].get("msg_id") == msg_id:
                    if msg["msg_type"] == "stream":
                        output = msg["content"]["text"]
                        results.append(CellResult(output))
                        cell_outputs.append(
                            new_output(
                                output_type="stream",
                                name=msg["content"]["name"],
                                text=output,
                            )
                        )
                    elif msg["msg_type"] == "execute_result":
                        output = msg["content"]["data"]["text/plain"]
                        results.append(CellResult(output))
                        cell_outputs.append(
                            new_output(
                                output_type="execute_result",
                                data=msg["content"]["data"],
                                execution_count=msg["content"]["execution_count"],
                            )
                        )
                    elif msg["msg_type"] == "error":
                        error = CellError(
                            ename=msg["content"]["ename"],
                            evalue=msg["content"]["evalue"],
                            traceback=msg["content"]["traceback"],
                        )
                        errors.append(error)
                        cell_outputs.append(
                            new_output(
                                output_type="error",
                                ename=msg["content"]["ename"],
                                evalue=msg["content"]["evalue"],
                                traceback=msg["content"]["traceback"],
                            )
                        )
                    elif (
                        msg["msg_type"] == "status"
                        and msg["content"]["execution_state"] == "idle"
                    ):
                        break
            except Exception as e:
                print(f"Error while executing cell: {e}")
                self.actions_on_error(code, cell_outputs)
                break

        if errors:
            self.actions_on_error(code, cell_outputs)
            raise errors[0]
        self.add_to_notebook(code, cell_outputs)

        return results

    def upload_file_env(self, file: IO, env_var: str = None):
        if not env_var:
            env_var = "MY_FILE"

        # Assuming we save the file and set the environment variable manually
        # file_path = "/tmp/" + file.name
        file_path =  file.name
        with open(file_path, "wb") as f:
            f.write(bytes(file.read(), "utf8"))

        # Setting the environment variable
        self.kc.execute(f"{env_var} = '{file_path}'")
        return env_var

    def add_to_notebook(self, code, outputs, notebook_path=None):
        """Saves the current IPython kernel state to an .ipynb file, including cell outputs."""
        if notebook_path is None:
            notebook_path = self.notebook_path

        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        cell = new_code_cell(code)
        cell.outputs = outputs
        nb.cells.append(cell)

        # Write the notebook to file
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        print(f"Notebook with outputs saved to {notebook_path}")

    def recreate_notebook(
        self,
    ):
        if not os.path.exists(self.notebook_path):
            # Create a new notebook
            nb = new_notebook()
            with open(self.notebook_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)

    def create_notebook(self, notebook_name: str, notebook_dir: str = "./notebooks"):

        # Create a subdirectory with the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        notebook_dir = os.path.join(notebook_dir, timestamp)
        os.makedirs(notebook_dir, exist_ok=True)

        notebook_path = os.path.join(notebook_dir, notebook_name)
        if not os.path.exists(notebook_path):
            # Create a new notebook
            nb = new_notebook()
            with open(notebook_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)
        return notebook_path


# if __name__ == "__main__":
#     url = "http://127.0.0.1:8888"
#     token = ""  # Replace with the actual token
#     jupyter_manager = JupyterExecutor(url, token)
#     jupyter_manager.test_and_execute("print('hello')")
#     jupyter_manager.test_and_execute("print('ss')")
#     jupyter_manager.test_and_execute("print(5*9)")
#     # jupyter_manager.add_to_notebook("my_notebook.ipynb", "./notebooks")
