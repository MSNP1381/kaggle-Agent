import json
import time
import httpx
from langgraph.graph import StateGraph, START, END
from code_generation_agent import CodeGenerationAgent
from nbexecutor import NBExecutor
from planner_agent import KaggleProblemPlanner
from replanner import KaggleProblemRePlanner
from executor_agent import KaggleCodeExecutor
from dotenv import load_dotenv
import os
from langfuse.callback import CallbackHandler
from states.main import KaggleProblemState
from task_enhancer import KaggleTaskEnhancer
from datautils import KaggleDataUtils
from langgraph.checkpoint.sqlite import SqliteSaver

# from loguru import logger
from datetime import datetime

# print)
# logfile = f"misc/logs/output{datetime.now().__str__()}.log"


class KaggleProblemSolver:
    def __init__(
        self,
        config,
        proxy,
    ):
        # super().__init__()
        self.config = config
        print(os.getenv("HTTP_PROXY_URL"))
        print("---" * 10)
        self.proxy = proxy
        self.nb_executor = NBExecutor()
        self.code_agent = CodeGenerationAgent(
            config, proxy=proxy, nb_executor=self.nb_executor
        )
        self.planner = KaggleProblemPlanner(
            config,
            proxy=proxy,
        )
        self.re_planner = KaggleProblemRePlanner(config, proxy=proxy)
        self.executor = KaggleCodeExecutor(self.nb_executor)
        self.enhancer = KaggleTaskEnhancer(config, proxy=proxy)
        self.data_utils = KaggleDataUtils(config, proxy)
        # self._init_state()

    def _init_state(self):
        self.dataset_path = "./house_prices.csv"
        self.problem_description = f"""
    Predict house prices based on various features.
    The evaluation metric is Root Mean Squared Error (RMSE).
    The dataset contains information about house features and their corresponding sale prices.
    dataset file name is : "{self.dataset_path}"
    """

        return {
            "problem_description": self.problem_description,
            "dataset_path": self.dataset_path,
        }

        # dataset_path = "house_prices.csv"  # Replace with actual path

    def is_plan_done(self, state: KaggleProblemState):
        # print("******************" * 2)
        # print("\n\n")
        # print("planned task no is:")
        # print(state.planned_tasks.__len__())
        # if state.planned_tasks.__len__() < 2 or True:
        #     print(*state.planned_tasks, sep="**\n\n")
        # print("\n\n")
        # print("******************" * 2)
        if not state.planned_tasks:
            return END
        return "enhancer"

    def compile(self,interrupt_before=None):
        graph_builder = StateGraph(KaggleProblemState)
        graph_builder.add_node("code_agent", self.code_agent)
        graph_builder.add_node("planner", self.planner)
        # graph_builder.add_node("re_planner", self.re_planner)
        graph_builder.add_node("executor", self.executor)
        graph_builder.add_node("enhancer", self.enhancer)
        graph_builder.add_node("data_utils", self.data_utils)

        graph_builder.add_edge(START, "data_utils")
        graph_builder.add_edge("data_utils", "planner")
        graph_builder.add_edge("planner", "enhancer")
        # graph_builder.add_conditional_edges("planner", self.is_plan_done)

        graph_builder.add_edge("enhancer", "code_agent")
        graph_builder.add_edge("code_agent", "executor")
        graph_builder.add_conditional_edges(
            "executor", self.is_plan_done, path_map={END: END, "enhancer": "enhancer"}
        )

        memory = SqliteSaver.from_conn_string(":memory:")

        self.graph = graph_builder.compile(
            checkpointer=memory, interrupt_before=interrupt_before
        )
        return self.graph

    def invoke(self, debug=True):
        state = self._init_state()
        for event in graph.stream(state,config=self.config):
            
            print(event)
        # self.graph.astream_log
        # return self.graph.invoke(state, config=self.config, debug=debug)

        #

    # def replan(self,state: KaggleProblemState):
    #

    #
    # def solve_problem(self, problem_description: str, dataset_path: str):
    #
    #     # Initial planning
    #     initial_plan = self.planner.plan(state)
    #     state.update_planned_tasks(initial_plan)
    #
    #     # Main execution loop
    #     while state.planned_tasks:
    #         task = state.planned_tasks.pop(0)
    #         state.update_task(task)
    #
    #         # Use the mediator to process the task
    #         result = self.mediator.process_task(task, state)
    #
    #         # Update state based on the result
    #         if 'code' in result:
    #             state.add_code(task, result['code'])
    #         if 'output' in result:
    #             state.add_result(task, result['output'])
    #
    #             # Update best score if applicable
    #             if isinstance(result['output'], dict) and 'score' in result['output']:
    #                 state.update_best_score(result['output']['score'])
    #
    #         # Replan after each task
    #         new_plan = self.replanner.replan(state)
    #         state.update_planned_tasks(new_plan)
    #
    #         print(f"Executed task: {task}")
    #         print(f"Updated plan: {state.planned_tasks}")
    #         print("---")
    #
    #     return state
    #


# Example usage
if __name__ == "__main__":
    
    print(".env loaded:", load_dotenv())

    proxy = httpx.Client(proxy=os.getenv("HTTP_PROXY_URL"))

    langfuse_handler = CallbackHandler(
        # httpx_client="",
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST"),
        session_id=f"session-{int(time.time())}",
    )
    # logger.add(logfile, colorize=True, enqueue=True,level=8)

    # handler_1 = FileCallbackHandler(logfile)
    # handler_2 = StdOutCallbackHandler()
    # r=KaggleProblemState()
    config = {
        "configurable": {"thread_id": str(int(time.time()))},
        "callbacks": [
            langfuse_handler,  # handler_1,handler_2
        ],
        "recursion_limit": 50,
    }
    solver = KaggleProblemSolver(config, proxy)
    graph = solver.compile()
    t = graph.get_graph(xray=2).draw_mermaid()
    # exit()
    res = solver.invoke()
    print(res)

    # problem_description = """
    # Predict house prices based on various features.
    # The evaluation metric is Root Mean Squared Error (RMSE).
    # The dataset contains information about house features and their corresponding sale prices.
    # data set file name is : "./house_prices.csv"
    # """
    # data_set_path = "./generated_notebooks/house_prices.csv"

    # dataset_path = "house_prices.csv"  # Replace with actual path

    # state_init = KaggleProblemState(problem_description=problem_description)

    # final_state = solver.solve_problem(problem_description, dataset_path)
    print(f"Best score achieved: {res.best_score}")
    print(f"Final model info: {res.model_info}")
