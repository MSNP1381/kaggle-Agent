import time
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
import copy
from nbconvert.preprocessors import CellExecutionError


class NBExecutor:
    def __init__(self):
        self.nb_name = ""
        self.nb: nbformat.NotebookNode = None

    def create_nb(self):
        self.nb_name = f"./generated_notebooks/notebook_{int(time.time())}.ipynb"
        nb = nbformat.v4.new_notebook()
        with open(self.nb_name, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        self.nb = nb
        return self.nb_name

    def add_nb_code_block(self, code):

        new_cell = nbformat.v4.new_code_cell(code)
        self.nb.cells.append(new_cell)
        with open(self.nb_name, "w") as f:
            nbformat.write(self.nb, f)

    def execute_notebook(self):
        ep = ExecutePreprocessor(timeout=600, kernel_name="myenv")
        ep.preprocess(
            self.nb,
            {
                "metadata": {
                    "kernelspec": {
                        "display_name": "myenv",
                        "language": "python",
                        "name": "python3",
                    },
                    "language_info": {"name": "python", "version": "3.12.3"},
                    "path": "./generated_notebooks",
                },
            },
        )

        # Save the executed notebook
        with open(self.nb_name, "w", encoding="utf8") as f:
            nbformat.write(self.nb, f)

    def test_and_execute(self, new_code):
        nb_ = copy.deepcopy(self.nb)
        # .ipynb length
        nb__name = self.nb_name[:-6] + f"_test_{int(time.time())}.ipynb"
        ep = ExecutePreprocessor(timeout=600, kernel_name="myenv")
        new_cell = nbformat.v4.new_code_cell(new_code)
        nb_.cells.append(new_cell)
        try:
            out = ep.preprocess(
                nb_,
                {
                    "metadata": {
                        "kernelspec": {
                            "display_name": "myenv",
                            "language": "python",
                            "name": "python3",
                        },
                        "language_info": {"name": "python", "version": "3.12.3"},
                        "path": "./generated_notebooks",
                    },
                },
            )
        except CellExecutionError as e:
            # print(e.ename)
            # print(e.evalue)
            # print(e.)
            # print(out)
            # msg = 'Error executing the notebook "%s".\n\n' % notebook_filename
            # msg += 'See notebook "%s" for the traceback.' % notebook_filename_out
            # print(msg)
            raise e

        finally:
            with open(nb__name, mode="w", encoding="utf-8") as f:
                nbformat.write(nb_, f)
        # return True

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

        # Find the last code cell with output
        for cell in reversed(self.nb.cells):
            if cell.cell_type == "code" and cell.outputs:
                return f"{self.process_cell_output(cell)}"

        return None  # If no output found


if __name__ == "__main__":
    e = NBExecutor()
    e.create_nb()
    e.test_and_execute("ptinynjvkdvnds")
    print("its ok")
