from autogen.coding import CodeBlock
from autogen.coding.base import IPythonCodeResult
from autogen.coding.jupyter import DockerJupyterServer, JupyterCodeExecutor
import nbformat
import time
import requests
import os


class CellException:
    s = 1


class NBExecutorAutoGen:
    def __init__(self, server):
        # self.server = DockerJupyterServer(
        #     # custom_image_name='kaggleagent_nb:latest'
        #     )
        self.server = server
        self.executor: JupyterCodeExecutor = None
        self.notebook_outputs = []
        self.code_blocks = []  # To keep track of added code blocks
        self.notebook = (
            nbformat.v4.new_notebook()
        )  # Initialize a new notebook for successful executions
        self.error_notebook = (
            nbformat.v4.new_notebook()
        )  # Initialize a new notebook for errors
        timestamp = int(time.time())
        self.notebook_name = (
            f"././generated_notebooks/generated_notebook_{timestamp}.ipynb"
        )
        self.error_notebook_name = (
            f"./generated_notebooks/error_notebook_{timestamp}.ipynb"
        )
        self.jupyter_base_url = (
            None  # Will store the Jupyter base URL after server start
        )
        self.executor = JupyterCodeExecutor(self.server)

    def upload_file_to_jupyter(self, local_file_path, remote_file_path=None):
        """
        Uploads a file to the Jupyter server.

        :param local_file_path: Path to the local file to be uploaded.
        :param remote_file_path: Path on the Jupyter server where the file will be uploaded.
        """
        if not self.jupyter_base_url:
            raise RuntimeError(
                "Jupyter server is not initialized. Call create_nb first."
            )

        if remote_file_path is None:
            remote_file_path = os.path.basename(local_file_path)

        # Prepare the file upload endpoint
        upload_url = f"{self.jupyter_base_url}files/{remote_file_path}"

        with open(local_file_path, "rb") as file_data:
            response = requests.put(upload_url, files={"file": file_data})

        if response.status_code == 201:
            print(f"File '{local_file_path}' successfully uploaded to '{upload_url}'")
        else:
            print(
                f"Failed to upload file '{local_file_path}'. Response code: {response.status_code}, Message: {response.text}"
            )

    def add_nb_code_block(self, code):
        if self.executor is None:
            raise RuntimeError(
                "JupyterCodeExecutor is not initialized. Call create_nb first."
            )

        # Execute the code block using the autogen executor
        code_block = CodeBlock(language="python", code=code)
        result = self.executor.execute_code_blocks([code_block])

        # Store the output and the code block
        self.notebook_outputs.append(result)
        self.code_blocks.append(code_block)
        self.add_code_cell_to_notebook(code, result)
        print(f"Executed code block: {code}\nOutput: {result}")

    def test_and_execute(self, new_code):
        if self.executor is None:
            raise RuntimeError(
                "JupyterCodeExecutor is not initialized. Call create_nb first."
            )

        print(f"Testing new code block: {new_code}")
        code_block = CodeBlock(language="python", code=new_code)

        # Attempt to execute the new code block
        result = self.executor.execute_code_blocks([code_block])
        if result.exit_code == 1:
            print(f"Error during execution: {str(result.output)}")
            self.log_error(new_code, str(result.output))
            print("Error logged successfully.")
        # If successful, store the code block and output
        self.notebook_outputs.append(result)
        self.code_blocks.append(code_block)
        self.add_code_cell_to_notebook(new_code, result)
        print(f"Test executed successfully: {new_code}\nOutput: {result}")

    def add_code_cell_to_notebook(self, code, result: IPythonCodeResult):
        # Create a new code cell
        new_cell = nbformat.v4.new_code_cell(code)

        # Set the output for the code cell
        output_data = {
            "output_type": "stream",
            "name": "stdout",
            "text": result.model_dump_json(indent=1),
        }
        new_cell["outputs"] = [
            nbformat.v4.new_output(
                output_type="stream",
                text=result.model_dump_json(indent=1),
            )
        ]

        # Append the cell to the notebook
        self.notebook.cells.append(new_cell)

        # Save the notebook
        with open(self.notebook_name, "w", encoding="utf-8") as f:
            nbformat.write(self.notebook, f, version=4)

    def log_error(self, code, error_message):
        # Create a new code cell for the error
        error_cell = nbformat.v4.new_code_cell(code)

        # Set the error output for the code cell
        error_output = {
            "output_type": "error",
            "ename": "ExecutionError",
            "evalue": error_message,
            "traceback": [error_message],
        }
        error_cell.outputs = [nbformat.v4.new_output(**error_output)]

        # Append the cell to the error notebook
        self.error_notebook.cells.append(error_cell)

        # Save the error notebook
        with open(self.error_notebook_name, "w", encoding="utf-8") as f:
            nbformat.write(self.error_notebook, f)
        return type("obj", (object,), error_output)

    def get_latest_output(self):
        # Return the output of the latest executed code block
        if not self.notebook_outputs:
            return "No outputs available."
        return self.notebook_outputs[-1]

    def execute_notebook(self):
        # Placeholder for full notebook execution
        # This would be more complex if you need to run a full sequence of code blocks
        print(
            "Executing full notebook is currently handled by executing code blocks one by one."
        )

    def cleanup(self):
        # Properly close the server
        self.server.__exit__(None, None, None)
        print("Jupyter server shutdown and resources cleaned up.")


# Example usage
if __name__ == "__main__":
    print("ss")
    with DockerJupyterServer() as server:
        e = NBExecutorAutoGen(server)
        print("ss22")
        time.sleep(2)
        # e.create_nb()
        e.add_nb_code_block("print('Hello, World!')")
        print("Latest output:", e.get_latest_output())

        # Test and execute a code block
        e.test_and_execute("print('This is a test block')")

        # Test a faulty code block to demonstrate error logging
        e.test_and_execute(
            "import non_existent_module"
        )  # This should cause an error and be logged

        print("Latest output:", e.get_latest_output())
        e.cleanup()
