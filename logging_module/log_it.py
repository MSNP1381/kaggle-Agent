from typing import Literal

from states.main import KaggleProblemState


class DictObj:
    def __init__(self, in_dict: dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(
                    self, key, [DictObj(x) if isinstance(x, dict) else x for x in val]
                )
            else:
                setattr(self, key, DictObj(val) if isinstance(val, dict) else val)


def log_it(
    state: "KaggleProblemState",
    type: Literal[
        "code_agent",
        "planner",
        "executor",
        "enhancer",
        "data_utils",
    ],
):
    # print(type)
    data_result = {}
    state = DictObj(state)
    match type:
        case "code_agent":
            data_result["context"] = {
                "problem_description": state.problem_description,
                "modelInfo": state.modelInfo,
                "planned_tasks": state.planned_tasks,
                "previous_tasks": state.get_task_results(),
                "current_task": state.enhanced_tasks[state.index],
            }
            data_result["output"] = state.task_codes_results[state.index]
        case "data_utils":
            data_result["output"] = {"dataset_info": state.dataset_info}
        case "enhancer":
            enhanced_task = state.enhanced_tasks[state.index]
            data_result["context"] = {
                "current_task": enhanced_task.task,
                "problem_description": state.problem_description,
                "dataset_info": state.dataset_info,
                "": state.get_task_results(),
                "planned_tasks": state.planned_tasks,
                "future_tasks": state.get_future_tasks(),
                "evaluation_metric": state.evaluation_metric,
                "best_score": state.best_score,
            }
            data_result["output"] = state.enhanced_tasks[-1]
        case "planner":
            data_result["context"] = {
                "problem_description": state.problem_description,
                "dataset_info": str(state.dataset_info),
            }
            data_result["output"] = {
                "planned_tasks": state.planned_tasks,
            }
    return data_result
