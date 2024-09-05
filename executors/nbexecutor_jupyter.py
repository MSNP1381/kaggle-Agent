from typing import IO, List
from jupyter_client import KernelManager
from utils import CellResult, NotebookExecutorInterface, CellError
import io


class JupyterExecutor(NotebookExecutorInterface):
    def __init__(self, url, token):
        # self.kc = kernel_client
        self.url = url
        self.token = token
        self.kernel()
        self.is_restarted = False

    def kernel(self):
        # url = "http://127.0.0.1:8888"
        # token = "393c18843508585c3d7d5a04290d4da8872b3cf2610f1edf"  # Replace with the actual token
        self.km = KernelManager(url=self.url, token=self.token)
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()

        # Wait until the kernel is ready
        self.kc.wait_for_ready()

    def reset(self):
        # Restart the kernel
        self.kc.restart_kernel()
        self.is_restarted = True

    def test_and_execute(self, code: str) -> List[CellResult]:
        results = []
        errors = []

        msg_id = self.kc.execute(code)
        while True:
            msg = self.kc.get_iopub_msg(timeout=5)
            if msg["parent_header"].get("msg_id") == msg_id:
                if msg["msg_type"] == "stream":
                    results.append(CellResult(msg["content"]["text"]))
                elif msg["msg_type"] == "execute_result":
                    results.append(CellResult(msg["content"]["data"]["text/plain"]))
                elif msg["msg_type"] == "error":
                    raise CellError(
                        **{
                            "ename": msg["content"]["ename"],
                            "evalue": msg["content"]["evalue"],
                            "traceback": msg["content"]["traceback"],
                        }
                    )
                elif (
                    msg["msg_type"] == "status"
                    and msg["content"]["execution_state"] == "idle"
                ):
                    # Execution is complete
                    break

        return results

    def upload_file_env(self, file: IO, env_var: str = None):
        if not env_var:
            env_var = "MY_FILE"

        # Assuming we save the file and set the environment variable manually
        file_path = "/tmp/" + file.name
        with open(file_path, "wb") as f:
            f.write(file.read())

        # Setting the environment variable
        # Assuming the kernel can access this file path (adjust depending on your environment)
        self.kc.execute(f"{env_var} = '{file_path}'")
        return env_var


if __name__ == "__main__":
    url = "http://127.0.0.1:8888"
    token = "8b36c92557afe1ff1d07e32df61492cac06ffbbb628f639f"  # Replace with the actual token
    jupyter_manager = JupyterExecutor(url, token)
    jupyter_manager.test_and_execute("print('hello')")
