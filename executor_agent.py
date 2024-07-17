import pandas as pd
from IPython.display import display, HTML
from nbexecutor import NBExecutor  # Assuming NBExecutor is in a file named nbexecutor.py

class KaggleCodeExecutor:
    def __init__(self):
        self.nb_executor = NBExecutor()
        self.nb_executor.create_nb()
        
        # Add initial imports
        initial_imports = """
import pandas as pd
from IPython.display import display, HTML
        """
        self.nb_executor.add_nb_code_block(initial_imports)
        self.nb_executor.execute_notebook()

    def execute_code(self, code):
        # Combine imports and code
        code_txt = code.imports + "\n" + code.code
        
        # Add the code to the notebook
        self.nb_executor.add_nb_code_block(code_txt)
        
        # Execute the notebook
        self.nb_executor.execute_notebook()
        
        # Get the latest output
        output = self.nb_executor.get_latest_output()
        
        return output if output else "#Code executed successfully\n"

    def get_dataframe(self, variable_name):
        # Add a code block to display the dataframe
        display_code = f"display({variable_name})"
        self.nb_executor.add_nb_code_block(display_code)
        self.nb_executor.execute_notebook()
        
        # Get the output (which should be the displayed dataframe)
        return self.nb_executor.get_latest_output()

    def get_variable(self, variable_name):
        # Add a code block to print the variable
        print_code = f"print({variable_name})"
        self.nb_executor.add_nb_code_block(print_code)
        self.nb_executor.execute_notebook()
        
        # Get the output (which should be the printed variable)
        return self.nb_executor.get_latest_output()