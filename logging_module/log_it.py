
from typing import Literal

import yaml

from states.main import KaggleProblemState


def log_it(state:'KaggleProblemState',type:Literal["CodeGenerationAgent","Plan","KaggleTaskEnhancer",'KaggleDataUtils','KaggleCodeExecutor']):
    d=state.dict()
    f_name=''
    match type:
        case "CodeGenerationAgent":
            d['task']=str(state.enhanced_task)   
            d["plans"]="\n".join(state.planned_tasks)   
            f_name='./misc/code_log.json' 
        case "Plan":
            ...
        case "KaggleTaskEnhancer":
            d.update(
                {"task": state.planned_tasks[0],
                # 'previous_tasks': str(state.previous_tasks),
                "task_results": state.get_task_results(),
                "planned_tasks": str(state.planned_tasks)})
            f_name='./misc/enhancer_log.json' 
            
        case "data":
            ...
        case "KaggleCodeExecutor":
            last_key=state.task_codes_results.keys()[-1]
            d['code']=state.task_codes_results[last_key][1]
            f_name='./misc/executor_log.json' 
            
    with open(f_name,'w') as f:
        yaml.dump(d,f)