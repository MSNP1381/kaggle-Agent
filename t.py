# File: agent.py

import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from planner import KaggleProblemPlanner
from replanner import KaggleProblemReplanner
from executor import KaggleCodeExecutor

@dataclass
class KaggleProblemState:
    problem_description: str
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    current_task: str = ""
    previous_tasks: List[str] = field(default_factory=list)
    task_codes: Dict[str, str] = field(default_factory=dict)
    task_results: Dict[str, Any] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)
    planned_tasks: List[str] = field(default_factory=list)
    evaluation_metric: Optional[str] = None
    best_score: Optional[float] = None

    def update_task(self, task: str):
        if self.current_task:
            self.previous_tasks.append(self.current_task)
        self.current_task = task

    def add_code(self, task: str, code: str):
        self.task_codes[task] = code

    def add_result(self, task: str, result: Any):
        self.task_results[task] = result

    def update_dataset_info(self, info: Dict[str, Any]):
        self.dataset_info.update(info)

    def update_model_info(self, info: Dict[str, Any]):
        self.model_info.update(info)

    def update_planned_tasks(self, tasks: List[str]):
        self.planned_tasks = tasks

    def update_best_score(self, score: float):
        if self.best_score is None or score > self.best_score:
            self.best_score = score

class KaggleProblemSolver:
    def __init__(self):
        self.planner = KaggleProblemPlanner()
        self.replanner = KaggleProblemReplanner()
        self.executor = KaggleCodeExecutor()

    def solve_problem(self, problem_description: str, dataset_path: str):
        state = KaggleProblemState(problem_description=problem_description)
        
        # Load dataset info
        df = pd.read_csv(dataset_path)
        state.update_dataset_info({
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
        })
        
        # Initial planning
        initial_plan = self.planner.plan(state)
        state.update_planned_tasks(initial_plan)
        
        # Main execution loop
        for task in state.planned_tasks:
            state.update_task(task)
            code = self.planner.generate_code(state)
            state.add_code(task, code)
            
            result = self.executor.execute_code(code)
            state.add_result(task, result)
            
            # Replan after each task
            new_plan = self.replanner.replan(state)
            state.update_planned_tasks(new_plan)
            
            print(f"Executed task: {task}")
            print(f"Updated plan: {state.planned_tasks}")
            print("---")
        
        return state

# Example usage
if __name__ == "__main__":
    solver = KaggleProblemSolver()
    problem_description = """
    Predict house prices based on various features.
    The evaluation metric is Root Mean Squared Error (RMSE).
    The dataset contains information about house features and their corresponding sale prices.
    """
    dataset_path = "house_prices.csv"  # Replace with actual path

    final_state = solver.solve_problem(problem_description, dataset_path)
    print(f"Best score achieved: {final_state.best_score}")
    print(f"Final model info: {final_state.model_info}")

# File: planner.py

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class KaggleProblemPlanner:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4")
        self.planner_prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant tasked with solving a Kaggle machine learning problem.
        Given the problem description and current state, create or update a plan to solve the problem.

        Problem Description:
        {problem_description}

        Current State:
        {state}

        Your task is to:
        1. Analyze the problem description and current state.
        2. Create or update a plan of tasks to solve the Kaggle problem.
        3. Ensure the plan covers all necessary steps: EDA, preprocessing, feature engineering, model selection, training, and evaluation.

        Respond with a list of planned tasks, each on a new line.
        """)
        
        self.code_generator_prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant tasked with generating Python code to solve a Kaggle machine learning problem.
        Given the current task and project state, generate the necessary code.

        Problem Description:
        {problem_description}

        Current Task: {current_task}

        Project State:
        {state}

        Your task is to:
        1. Generate Python code to accomplish the current task.
        2. Ensure the code is compatible with the Kaggle notebook environment.
        3. Use pandas, scikit-learn, and other common ML libraries as needed.
        4. Include comments to explain key steps.

        Respond with only the Python code, without any additional explanation.
        """)

    def plan(self, state):
        response = self.llm.invoke(self.planner_prompt.format_messages(
            problem_description=state.problem_description,
            state=str(state.__dict__)
        ))
        return response.content.strip().split("\n")

    def generate_code(self, state):
        response = self.llm.invoke(self.code_generator_prompt.format_messages(
            problem_description=state.problem_description,
            current_task=state.current_task,
            state=str(state.__dict__)
        ))
        return response.content.strip()

# File: replanner.py

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class KaggleProblemReplanner:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4")
        self.replan_prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant tasked with replanning a Kaggle machine learning project based on the latest execution results.
        Given the current state of the project and the output of the last executed task, determine if the plan needs to be adjusted.

        Problem Description:
        {problem_description}

        Current State:
        {state}

        Last executed task: {last_task}
        Execution result: {execution_result}

        Current plan:
        {current_plan}

        Your task is to:
        1. Analyze the execution result and determine if it requires changes to the plan.
        2. If changes are needed, provide a new list of planned tasks.
        3. If no changes are needed, return the current plan.

        Respond with a list of planned tasks, each on a new line.
        """)

    def replan(self, state):
        last_task = state.current_task
        execution_result = state.task_results.get(last_task, "No result available")

        response = self.llm.invoke(self.replan_prompt.format_messages(
            problem_description=state.problem_description,
            state=str(state.__dict__),
            last_task=last_task,
            execution_result=execution_result,
            current_plan="\n".join(state.planned_tasks)
        ))

        return response.content.strip().split("\n")

# File: executor.py

import pandas as pd
from IPython.display import display, HTML

class KaggleCodeExecutor:
    def execute_code(self, code: str) -> Any:
        # In a real implementation, this would use the Kaggle API or a Jupyter notebook
        # For this example, we'll use a simple exec with a custom global namespace
        globals_dict = {
            'pd': pd,
            'display': display,
            'HTML': HTML,
            # Add other necessary libraries here
        }
        exec(code, globals_dict)
        return globals_dict.get('result', "Code executed successfully")