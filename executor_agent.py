import pandas as pd
from IPython.display import display, HTML
from nbexecutor import (
    NBExecutor,
)  # Assuming NBExecutor is in a file named nbexecutor.py
from states.main import KaggleProblemState


class KaggleCodeExecutor:
    def __init__(self, nb_executor: NBExecutor):
        self.nb_executor = nb_executor
        self.nb_executor.create_nb()

        # Add initial imports
        initial_imports = (
            """import pandas as pd\nfrom IPython.display import display, HTML"""
        )
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
        display_code = f'print("""\n{variable_name}\n""")'
        self.nb_executor.add_nb_code_block(display_code)
        self.nb_executor.execute_notebook()

        # Get the output (which should be the displayed dataframe)
        return self.nb_executor.get_latest_output()

    def get_variable(self, variable_name):
        # Add a code block to print the variable
        print_code = f'print("""\n{variable_name}""")'
        self.nb_executor.add_nb_code_block(print_code)
        self.nb_executor.execute_notebook()

        # Get the output (which should be the printed variable)
        return self.nb_executor.get_latest_output()

    def __call__(self, state: KaggleProblemState):
        enhanced_task = state.enhanced_task
        code = state.task_codes_results[enhanced_task.task][0]
        output = self.execute_code(code)

        if enhanced_task.requires_code_output:

            # output = self.executor.execute_code(code)

            if enhanced_task.expected_output_type == "dataframe":
                output = self.get_dataframe(output)
            elif enhanced_task.expected_output_type in ["plot", "metric", "model"]:
                output = self.get_variable(output)

        state.task_codes_results[enhanced_task.task] = (code, str(output))

        return {"task_codes_results": state.task_codes_results}
