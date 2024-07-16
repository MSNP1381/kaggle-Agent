import time
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor


class NBExecutor:
    def __init__(self):
        self.nb_name = ""

    def create_nb(self):
        self.nb_name = f"./generated_notebooks/notebook_{int(time.time())}.ipynb"
        nb = nbformat.v4.new_notebook()
        with open(self.nb_name, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        return self.nb_name

    def add_nb_code_block(self, code):
        with open(self.nb_name, "r", encoding="utf8") as f:
            nb: nbformat.NotebookNode = nbformat.read(f, 4)
        new_cell = nbformat.v4.new_code_cell(code)
        nb.cells.append(new_cell)
        with open(self.nb_name, "w") as f:
            nbformat.write(nb, f)

    def execute_notebook(self):
        with open(self.nb_name, "r", encoding="utf8") as f:
            nb: nbformat.NotebookNode = nbformat.read(f, 4)

        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        ep.preprocess(
            nb,
            {
                "metadata": {
                    "kernelspec": {
                        "display_name": ".venv",
                        "language": "python",
                        "name": "python3",
                    },
                    "language_info": {"name": "python", "version": "3.12.3"},
                },
                "path": "./generated_notebooks",
            },
        )

        # Save the executed notebook
        with open(self.nb_name, "w", encoding="utf8") as f:
            nbformat.write(nb, f)

    def process_cell_output(self, cell):
        if cell.cell_type != "code" or not cell.outputs:
            return None

        for output in cell.outputs:
            if output.output_type == "execute_result":
                return output.data.get("text/plain")
            elif output.output_type == "stream":
                return output.text
            elif output.output_type == "display_data":
                return output.data.get("text/plain") or str(output.data)
            elif output.output_type == "error":
                return f"Error: {output.ename}\n{output.evalue}"

        return None

    def get_latest_output(self):
        with open(self.nb_name, "r", encoding="utf8") as f:
            nb: nbformat.NotebookNode = nbformat.read(f, 4)

        # Find the last code cell with output
        for cell in reversed(nb.cells):
            if cell.cell_type == "code" and cell.outputs:
                return self.process_cell_output(cell)

        return None  # If no output found
