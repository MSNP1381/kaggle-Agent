from injector import inject
from code_generation_agent import CodeGenerationAgent
from states.main import KaggleProblemState
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import NotebookFailError


class KaggleCodeManager:
    @inject
    def __init__(self, code_agent: CodeGenerationAgent):
        self.code_agent = code_agent
        code_agent.max_iterations = 1
        self.temp = 0

    def run(self, state):
            temp_increase = 0.65
            for i in range(3):
                try:

                    return self.code_agent(
                        state=state, temp=self.temp, max_iterations=1 if i < 1 else 2
                    )

                except NotebookFailError as e:
                    logging.error(f"Error in code agent execution: {e}")
                    self.temp += temp_increase
                    self.temp = min(2.0, self.temp)
                except Exception as e:
                    raise e  # Raise error if it is not NotebookFailError
                
                
            raise Exception("We are Cooked!!!!\n\n")

    # def run_threaded(self, state):

    def __call__(self, state: KaggleProblemState):
        return self.run(state)
