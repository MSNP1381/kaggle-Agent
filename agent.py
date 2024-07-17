    import time
    import httpx
    import pandas as pd
    from typing import Dict, Any, List, Optional
    from dataclasses import dataclass, field
    from planner_agent import KaggleProblemPlanner
    from replanner import KaggleProblemReplanner
    from executor_agent import KaggleCodeExecutor
    from dotenv import load_dotenv
    from langfuse.callback import CallbackHandler
    from task_enhancer import KaggleTaskEnhancer
    from task_mediator import KaggleTaskMediator
    from dataclasses import dataclass, field
    from typing import Dict, Any, List, Optional

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

        def set_evaluation_metric(self, metric: str):
            self.evaluation_metric = metric
    class KaggleProblemSolver:
        def __init__(self, config):
            self.config = config
            proxy = httpx.Client(proxy="http://127.0.0.1:2081")
            self.planner = KaggleProblemPlanner(config,proxy=proxy)
            self.replanner = KaggleProblemReplanner(config,proxy=proxy)
            self.executor = KaggleCodeExecutor()
            self.enhancer = KaggleTaskEnhancer(config,proxy=proxy)
            self.mediator = KaggleTaskMediator(config, self.planner, self.executor, self.enhancer)

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
            while state.planned_tasks:
                task = state.planned_tasks.pop(0)
                state.update_task(task)
                
                # Use the mediator to process the task
                result = self.mediator.process_task(task, state)
                
                # Update state based on the result
                if 'code' in result:
                    state.add_code(task, result['code'])
                if 'output' in result:
                    state.add_result(task, result['output'])
                    
                    # Update best score if applicable
                    if isinstance(result['output'], dict) and 'score' in result['output']:
                        state.update_best_score(result['output']['score'])
                
                # Replan after each task
                new_plan = self.replanner.replan(state)
                state.update_planned_tasks(new_plan)
                
                print(f"Executed task: {task}")
                print(f"Updated plan: {state.planned_tasks}")
                print("---")
            
            return state

    # Example usage
    if __name__ == "__main__":
        print(".env loaded:", load_dotenv())
        langfuse_handler = CallbackHandler(
            public_key="pk-lf-d578c2a8-4208-485a-b0ab-d72cca7fdeac",
            secret_key="sk-lf-16e12bce-ad9e-43c5-b06a-7c1fb1f67f7f",
            host="http://127.0.0.1:3000",
            session_id=f"session-{int(time.time())}"
        )
        
        config = {"callbacks": [langfuse_handler]}
        solver = KaggleProblemSolver(config)
        problem_description = """
        Predict house prices based on various features.
        The evaluation metric is Root Mean Squared Error (RMSE).
        The dataset contains information about house features and their corresponding sale prices.
        data set file name is : "./house_prices.csv"
        """
        dataset_path = "house_prices.csv"  # Replace with actual path

        final_state = solver.solve_problem(problem_description, dataset_path)
        print(f"Best score achieved: {final_state.best_score}")
        print(f"Final model info: {final_state.model_info}")