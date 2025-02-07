import logging
import os
import shutil
import time
from datetime import datetime
from typing import List

import nbformat
from jupyter_client import KernelManager
from nbformat.v4 import new_code_cell, new_notebook, new_output

from utils import CellError, CellResult, NotebookExecutorInterface
from jupyter_client import KernelManager

# Set up logging
logger = logging.getLogger(__name__)


class JupyterExecutor(NotebookExecutorInterface):
    def __init__(self, url, token):
        self.url = url
        self.token = token
        self.kernel()
        self.is_restarted = False
        self.notebook_path = self.create_nb("my_notebook.ipynb", "./notebooks")
        logger.info(
            f"JupyterExecutor initialized with notebook path: {self.notebook_path}"
        )

    def kernel(self):
        logger.info("Connecting to remote kernel...")

        # Create and start a local kernel using KernelManager.
        km = KernelManager(kernel_name="python3")
        km.start_kernel()
        self.kc = km.client()

        # Start channel communications and wait for the kernel to become ready.
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=60)
        logger.info("Local kernel started.")

    def reset(self):
        logger.info("Resetting kernel...")
        self.kc.execute("%reset -f")
        self.is_restarted = True
        logger.info("Kernel reset.")
        self.kc.wait_for_ready()
        time.sleep(2)  # Give a short pause to ensure kernel is fully ready
        logger.info("Kernel is ready.")

    def actions_on_error(self, code, cell_outputs):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        error_notebook_path = os.path.join(
            os.path.dirname(self.notebook_path),
            "error_" + timestamp + "_" + os.path.basename(self.notebook_path),
        )
        shutil.copy(self.notebook_path, error_notebook_path)
        self.add_to_notebook(code, cell_outputs, error_notebook_path)
        logger.error(f"Error occurred. Error notebook saved at: {error_notebook_path}")

    def clean_code(self, code):
        return code.strip().replace("\\n", "\n")

    def test_and_execute(
        self,
        code: str,
        notebook_name: str = "my_notebook.ipynb",
        notebook_dir: str = "./notebooks",
    ) -> List[CellResult]:
        if self.kc is None:
            logger.warning("Kernel client is None. Restarting kernel.")
            self.restart_kernel()

        results = None
        errors = []

        logger.info(
            f"Executing code: {code[:50]}..."
        )  # Log first 50 characters of code
        msg_id = self.kc.execute(code)
        cell_outputs = []

        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=180)
                if msg["parent_header"].get("msg_id") == msg_id:
                    if msg["msg_type"] == "stream":
                        output = msg["content"]["text"]
                        results = CellResult(output)
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
                        logger.debug(
                            f"Execute result: {output[:50]}..."
                        )  # Log first 50 characters of output
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
                        logger.error(
                            f"Error in cell execution: {error.ename} - {error.evalue}"
                        )
                    elif (
                        msg["msg_type"] == "status"
                        and msg["content"]["execution_state"] == "idle"
                    ):
                        break
            except Exception as e:
                logger.exception(f"Error while executing cell: {e}")
                self.actions_on_error(code, cell_outputs)
                break

        if errors:
            self.actions_on_error(code, cell_outputs)
            raise errors[0]
        self.add_to_notebook(
            code,
            cell_outputs,
        )
        self.verify_notebook()
        logger.info(f"Execution completed. Results: {results}")
        return results

    def add_to_notebook(self, code, outputs, notebook_path=None):
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

        logger.info(f"Notebook saved. Total cells: {len(nb.cells)}")

    def verify_notebook(self, notebook_path=None):
        if notebook_path is None:
            notebook_path = self.notebook_path

        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        logger.info(f"Verifying notebook: {notebook_path}")
        logger.info(f"Total cells: {len(nb.cells)}")

    def create_nb(self, notebook_name: str, notebook_dir: str = "./notebooks"):
        # Create a subdirectory with the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        notebook_dir = os.path.join(notebook_dir, timestamp)
        os.makedirs(notebook_dir, exist_ok=True)

        notebook_path = os.path.join(notebook_dir, notebook_name)
        if not os.path.exists(notebook_path):
            # Create a new notebook
            nb = new_notebook()
            with open(notebook_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f, 4)
        logger.info(f"Created new notebook: {notebook_path}")
        return notebook_path


# Commented out main block
# if __name__ == "__main__":
#     url = "http://127.0.0.1:8888"
#     token = ""  # Replace with the actual token
#     jupyter_manager = JupyterExecutor(url, token)
#     jupyter_manager.test_and_execute("print('hello')")
#     jupyter_manager.test_and_execute("print('ss')")
#     jupyter_manager.test_and_execute("print(5*9)")
#     # jupyter_manager.add_to_notebook("my_notebook.ipynb", "./notebooks")
