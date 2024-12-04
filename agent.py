#!/home/msnp/miniconda3/envs/kaggle_agent/bin/python
import argparse
import os
import time

import kaggle
from dotenv import load_dotenv
from injector import inject
from langgraph.graph import END, START, StateGraph
from pymongo import MongoClient
from data_utils import DataUtils
from di_container import create_injector
from executors.nbexecutor_jupyter import JupyterExecutor
from kaggle_scraper import ScrapeKaggle
from persistence.mongo import MongoDBSaver

# from planner_agent import KaggleProblemPlanner
from planner_agent import KaggleProblemPlanner
from code_generation_agent import CodeGenerationAgent

# from langfuse.callback import CallbackHandler
from states.main import KaggleProblemState
from task_enhancer import KaggleTaskEnhancer


class KaggleProblemSolver:
    @inject
    def __init__(
        self,
        config: dict,
        # proxy: httpx.Client,
        client: MongoClient,
        nb_executor: JupyterExecutor,
        scraper: ScrapeKaggle,
        planner: KaggleProblemPlanner,
        # re_planner: KaggleProblemRePlanner,
        code_manager: CodeGenerationAgent,
        enhancer: KaggleTaskEnhancer,
        data_utils: DataUtils,
        # handler: CallbackHandler,
    ):
        self.config = config
        # self.proxy = proxy
        self.client = client
        self.nb_executor = nb_executor
        self.scraper = scraper
        self.planner = planner
        # self.re_planner = re_planner
        self.code_manager = code_manager
        self.enhancer = enhancer
        self.data_utils = data_utils
        # self.handler = handler

    def _init_state(self, url: str):
        self.dataset_path = "./input/train.csv"
        self.test_dataset_path = "./input/test.csv"

        return KaggleProblemState(
            **{
                # "file_env_var": env_var,
                "challenge_url": url,
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
            # return END
            return END
        return "enhancer"

    def compile(self, checkpointer, interrupt_before=None):
        graph_builder = StateGraph(KaggleProblemState)
        graph_builder.add_node("scraper", self.scraper)
        graph_builder.add_node("code_manager", self.code_manager)
        graph_builder.add_node("planner", self.planner)
        # graph_builder.add_node("re_planner", self.re_planner)
        # graph_builder.add_node("executor", self.executor)
        graph_builder.add_node("enhancer", self.enhancer)
        graph_builder.add_node("data_utils", self.data_utils)

        graph_builder.add_edge(START, "scraper")
        graph_builder.add_edge("scraper", "data_utils")
        graph_builder.add_edge("data_utils", "planner")
        graph_builder.add_edge("planner", "enhancer")
        # graph_builder.add_conditional_edges("planner", self.is_plan_done)

        graph_builder.add_edge("enhancer", "code_manager")
        # graph_builder.add_edge("code_agent", "en")  # Connect code_agent to submission_node
        # graph_builder.add_edge("submission_node", "executor")
        graph_builder.add_conditional_edges(
            "code_manager",
            self.is_plan_done,
            path_map={"enhancer": "enhancer", END: END},
        )

        # memory = SqliteSaver.from_conn_string(":memory:")

        self.graph = graph_builder.compile(
            checkpointer=checkpointer, interrupt_before=interrupt_before
        )
        return self.graph

    def invoke(self, url: str, thread_id, debug=False):
        state = self._init_state(url)
        # self.config["configurable"]["thread_id"] = thread_id
        return self.graph.invoke(state, config=self.config, debug=debug)

    def submit_to_kaggle(self, competition: str, submission_file: str, message: str):
        """
        Submits the specified file to the given Kaggle competition and retrieves the result.

        :param competition: The name of the Kaggle competition.
        :param submission_file: The path to the submission file.
        :param message: The submission message.
        :return: The result of the submission.
        """
        # Submit the file to the competition
        print(f"Submitting {submission_file} to {competition} with message: {message}")
        kaggle.api.competition_submit(submission_file, message, competition)

        # Retrieve the submission result
        submissions = kaggle.api.competition_submissions(competition)
        latest_submission = max(submissions, key=lambda x: x["date"])
        print(f"Latest submission result: {latest_submission}")

        return latest_submission


# Example usage
if __name__ == "__main__":
    print(".env loaded:", load_dotenv(override=True))

    parser = argparse.ArgumentParser("kaggle_scraper")
    parser.add_argument(
        "--url",
        help="url to challenge",
        type=str,
        required=False,
        default="https://www.kaggle.com/competitions/nlp-getting-started/",
    )
    parser.add_argument(
        "--cached",
        help="use cached version",
        type=bool,
        required=False,
    )
    args = parser.parse_args()
    # proxy = httpx.Client(proxy=os.getenv("HTTP_PROXY_URL"))
    proxy = None
    session_id = f"session-{int(time.time())}"
    # langfuse_handler = CallbackHandler(
    #     public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    #     secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    #     host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    #     session_id=session_id,
    # )

    client = MongoClient(
        host=os.getenv("MONGO_HOST"), port=int(os.getenv("MONGO_PORT"))
    )
    checkpointer = MongoDBSaver(client, db_name="checkpoints")

    config = {
        "configurable": {"thread_id": session_id},
        # "callbacks": [
        #     langfuse_handler,  # handler_1,handler_2
        # ],
        "recursion_limit": 50,
        "checkpointer": checkpointer,
    }
    injector, app_module = create_injector()
    solver = injector.get(KaggleProblemSolver)
    graph = solver.compile(checkpointer)
    result = solver.invoke(args.url, debug=args.cached)

    # Configuration for submission
    submission_config = {
        "competition": "nlp-getting-started",
        "submission_file": "path/to/your/submission.csv",
        "submission_message": "My first submission",
    }

    # Invoke the submission node
    state = result  # Assuming `invoke` returns the current state
    submission_result = solver.submission_node.run(state, submission_config)
    print(submission_result)
