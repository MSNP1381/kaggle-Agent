
class KaggleTaskMediator:
    def __init__(self, config, planner, executor, enhancer):
        self.config = config
        self.planner = planner
        self.executor = executor
        self.enhancer = enhancer

    def process_task(self, task, state):
        enhanced_task = self.enhancer.enhance_task(task, state)
        
        if enhanced_task.requires_code_output:
            code = self.planner.generate_code(state)
            output = self.executor.execute_code(code)
            
            if enhanced_task.expected_output_type == 'dataframe':
                output = self.executor.get_dataframe(output)
            elif enhanced_task.expected_output_type in ['plot', 'metric', 'model']:
                output = self.executor.get_variable(output)
            
            return {
                'task': enhanced_task.task,
                'description': enhanced_task.enhanced_description,
                'code': code,
                'output': output
            }
        else:
            return {
                'task': enhanced_task.task,
                'description': enhanced_task.enhanced_description
            }

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