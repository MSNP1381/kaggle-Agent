import argparse
import json
import time
import httpx
from autogen.coding.jupyter import DockerJupyterServer
from langgraph.graph import StateGraph, START, END
from psycopg_pool import ConnectionPool
from pymongo import MongoClient

from code_generation_agent import CodeGenerationAgent, CodeGraphState

# from NBExecutorAutoGen import JupyterNBExecutor
from kaggle_scaper import ScrapeKaggle
from nbexecutor_autoGen import NBExecutorAutoGen
from nbexecutor import NBExecutor
from persistence.mongo import MongoDBSaver

from planner_agent import KaggleProblemPlanner
from replanner import KaggleProblemRePlanner
from executor_agent import KaggleCodeExecutor
from dotenv import load_dotenv
import os
from langfuse.callback import CallbackHandler
from states.main import KaggleProblemState
from task_enhancer import KaggleTaskEnhancer
from dataUtils_agent import KaggleDataUtils

# from loguru import logger
from datetime import datetime


# print)
# logfile = f"misc/logs/output{datetime.now().__str__()}.log"


class KaggleProblemSolver:
    def __init__(self, config, proxy, client,server, args):
        self.url = args.url
        # super().__init__()
        self.config = config
        self.client = client
        print(os.getenv("HTTP_PROXY_URL"))
        print("---" * 10)
        self.proxy = proxy
        # self.nb_executor = NBExecutor()

        self.nb_executor = NBExecutorAutoGen(server)
        self.code_agent = CodeGenerationAgent(
            config, proxy=proxy, nb_executor=self.nb_executor
        )
        self.scraper = ScrapeKaggle(self.client, self.config)
        self.planner = KaggleProblemPlanner(config, proxy=proxy)
        self.re_planner = KaggleProblemRePlanner(config, proxy=proxy)
        self.executor = KaggleCodeExecutor(self.nb_executor)
        self.enhancer = KaggleTaskEnhancer(config, proxy=proxy)
        self.data_utils = KaggleDataUtils(config, proxy)
        # self._init_state()

    def _init_state(self):
        self.dataset_path = "./train.csv"
        # self.nb_executor.create_nb()
        self.nb_executor.upload_file_to_jupyter(self.dataset_path)
        return KaggleProblemState(
            **{
                "challenge_url": self.url,
                "dataset_path": self.dataset_path,
            }
        )

    def is_plan_done(self, state: KaggleProblemState):
        # print("******************" * 2)
        # print("\n\n")
        # print("planned task no is:")
        # print(state.planned_tasks.__len__())
        # if state.planned_tasks.__len__() < 2 or True:
        #     print(*state.planned_tasks, sep="**\n\n")
        # print("\n\n")
        # print("******************" * 2)
        print(
            "********tot_Iter**********\n\n         ",
            state.index + 1,
            "/",
            len(state.planned_tasks),
            "\n\n**************************",
        )
        if state.index == len(state.planned_tasks) - 1:
            return END
        return "enhancer"

    def compile(self, checkpointer):
        graph_builder = StateGraph(KaggleProblemState)
        graph_builder.add_node("scraper", self.scraper)
        graph_builder.add_node("code_agent", self.code_agent)
        graph_builder.add_node("planner", self.planner)
        # graph_builder.add_node("re_planner", self.re_planner)
        graph_builder.add_node("executor", self.executor)
        graph_builder.add_node("enhancer", self.enhancer)
        graph_builder.add_node("data_utils", self.data_utils)

        graph_builder.add_edge(START, "scraper")
        graph_builder.add_edge("scraper", "data_utils")
        graph_builder.add_edge("data_utils", "planner")
        graph_builder.add_edge("planner", "enhancer")
        # graph_builder.add_conditional_edges("planner", self.is_plan_done)

        graph_builder.add_edge("enhancer", "code_agent")
        graph_builder.add_edge("code_agent", "executor")
        graph_builder.add_conditional_edges(
            "executor", self.is_plan_done, path_map={END: END, "enhancer": "enhancer"}
        )

        # memory = SqliteSaver.from_conn_string(":memory:")

        self.graph = graph_builder.compile(checkpointer=checkpointer)
        return self.graph

    def invoke(self, debug=True):
        state = self._init_state()
        return graph.invoke(state, config=self.config)


# Example usage
if __name__ == "__main__":
    print(".env loaded:", load_dotenv())
    parser = argparse.ArgumentParser("kaggle_scraper")
    parser.add_argument(
        "--url", help="url to challenge", type=str, required=True, default="https://www.kaggle.com/competitions/nlp-getting-started/"
    )
    parser.add_argument(
        "--cached",
        help="use cached version",
        type=bool,
        required=False,
    )
    args = parser.parse_args()
    proxy = httpx.Client(proxy=os.getenv("HTTP_PROXY_URL"))

    langfuse_handler = CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST"),
        session_id=f"session-{int(time.time())}",

    )

    client = MongoClient(
        host=os.getenv("MONGO_HOST"), port=int(os.getenv("MONGO_PORT"))
    )
    checkpointer = MongoDBSaver(client, db_name="checkpoints")

    config = {
        "configurable": {"thread_id": str(int(time.time()))},
        # "callbacks": [
        #     langfuse_handler,  # handler_1,handler_2
        # ],
        "recursion_limit": 50,
        "checkpointer": checkpointer,
    }
    with DockerJupyterServer() as server:
        solver = KaggleProblemSolver(config, proxy, client, server,args)
    graph = solver.compile(checkpointer)
    # exit()
    solver.invoke()

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
    # print(f"Best score achieved: {res.best_score}")
    # print(f"Final model info: {res.model_info}")
