import requests
import json
import time
import os
from urllib.parse import urljoin

class JupyterServerExecutor:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Token {token}',
            'Content-Type': 'application/json'
        }

    def create_new_notebook(self, path):
        url = urljoin(self.base_url, 'api/contents')
        data = {
            'type': 'notebook',
            'path': path
        }
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()

    def update_notebook(self, path, cells):
        url = urljoin(self.base_url, f'api/contents/{path}')
        response = requests.get(url, headers=self.headers)
        notebook = response.json()
        
        notebook['content']['cells'] = cells
        
        response = requests.put(url, headers=self.headers, json=notebook)
        return response.json()

    def execute_notebook(self, path):
        kernels_url = urljoin(self.base_url, 'api/kernels')
        response = requests.post(kernels_url, headers=self.headers)
        kernel_id = response.json()['id']

        execute_url = urljoin(self.base_url, f'api/kernels/{kernel_id}/channels')
        
        websocket_url = f"ws://localhost:8888/api/kernels/{kernel_id}/channels"
        print(f"WebSocket URL: {websocket_url}")
        print("Please connect to this WebSocket to receive real-time execution results.")

        # Here, you would typically connect to the WebSocket and handle messages
        # For simplicity, we'll use a synchronous approach
        
        cells_url = urljoin(self.base_url, f'api/contents/{path}')
        response = requests.get(cells_url, headers=self.headers)
        notebook = response.json()
        
        for cell in notebook['content']['cells']:
            if cell['cell_type'] == 'code':
                execute_request = {
                    'header': {
                        'msg_id': 'execute',
                        'username': 'test',
                        'session': '1',
                        'msg_type': 'execute_request',
                        'version': '5.2'
                    },
                    'content': {
                        'code': cell['source'],
                        'silent': False,
                        'store_history': True,
                        'user_expressions': {},
                        'allow_stdin': False
                    },
                    'parent_header': {},
                    'metadata': {},
                    'buffers': []
                }
                response = requests.post(execute_url, headers=self.headers, json=execute_request)
                print(f"Execution response: {response.text}")
                
                # In a real implementation, you'd process the WebSocket messages here
                time.sleep(2)  # Simulating execution time

        # Delete the kernel when done
        delete_url = urljoin(self.base_url, f'api/kernels/{kernel_id}')
        requests.delete(delete_url, headers=self.headers)

    def get_notebook_content(self, path):
        url = urljoin(self.base_url, f'api/contents/{path}')
        response = requests.get(url, headers=self.headers)
        return response.json()

# Usage example
if __name__ == "__main__":
    base_url = "http://localhost:8888"  # Update this to your Jupyter server URL
    token = "your_token_here"  # Update this with your actual token
    
    executor = JupyterServerExecutor(base_url, token)
    
    # Create a new notebook
    new_notebook = executor.create_new_notebook("LLMGeneratedCode.ipynb")
    
    # Update the notebook with LLM-generated code
    cells = [
        {
            "cell_type": "code",
            "source": "import pandas as pd\n\n# Load the data\ndf = pd.read_csv('kaggle_dataset.csv')\nprint(df.head())"
        },
        {
            "cell_type": "code",
            "source": "# Perform some analysis\nprint(df.describe())"
        }
    ]
    executor.update_notebook("LLMGeneratedCode.ipynb", cells)
    
    # Execute the notebook
    executor.execute_notebook("LLMGeneratedCode.ipynb")
    
    # Get the results
    results = executor.get_notebook_content("LLMGeneratedCode.ipynb")
    print(json.dumps(results, indent=2))
