import subprocess
class VenvExecutionError(Exception):
    """Custom exception for virtual environment execution errors."""
    pass

def exec_in_venv(code, venv_path="./.venv"):
    # Construct the Python command
    python_executable = f"{venv_path}/bin/python"
    
    # Use subprocess to run the code in the specified virtual environment
    process = subprocess.Popen([python_executable, '-c', code],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True)
    
    # Capture output and errors
    stdout, stderr = process.communicate()
    
    # Check if the process executed successfully
    if process.returncode != 0:
        error_message = f"Execution failed with return code {process.returncode}.\nStderr: {stderr}"
        raise VenvExecutionError(error_message)
    
    return stdout

# # Example usage
# venv_path = "./.venv"

# stdout, stderr, returncode = exec_in_venv(code_to_execute, venv_path)

# print(f"Output: {stdout}")
# print(f"Errors: {stderr}")
# print(f"Return code: {returncode}")