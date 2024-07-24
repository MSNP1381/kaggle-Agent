from executor_agent import KaggleCodeExecutor
from planner_agent import KaggleProblemPlanner
from task_enhancer import KaggleTaskEnhancer
# from agent import KaggleProblemState

class KaggleTaskMediator:
    def __init__(self, config, planner:KaggleProblemPlanner, executor:KaggleCodeExecutor, enhancer:KaggleTaskEnhancer):
        self.config = config
        self.planner = planner
        self.executor = executor
        self.enhancer = enhancer



    def run_pipeline(self, state):
        plan = self.planner.plan(state)
        results = []
        
        for task in plan:
            result = self.process_task(task, state)
            results.append(result)
            
            # Update state based on task result
            if 'output' in result:
                # You might want to update the state in a more sophisticated way
                state.last_output = result['output']
            
            state.current_task = task
        
        return results