# di_container.py
from injector import Injector, Module, singleton, provider
import httpx
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from code_generation_agent import CodeGenerationAgent
from kaggle_scraper import ScrapeKaggle
from psycopg import Connection
# from nbexecuter_e2b import E2B_executor, SandboxManager
from executors.nbexecutor_jupyter import JupyterExecutor
from planner_agent import KaggleProblemPlanner
from replanner import KaggleProblemRePlanner
from task_enhancer import KaggleTaskEnhancer
from data_utils import DataUtils
import os
from config_reader import config_reader

from langfuse.callback import CallbackHandler
import time

from submission.submission import SubmissionNode  # Correct Python import

class AppModule(Module):

    def __init__(self):
        self.server = None

    @singleton
    @provider
    def provide_config(self) -> dict:
        session_id = f"session-{int(time.time())}"
        return {
            "configurable": {"thread_id": str(int(time.time()))},
            "recursion_limit": config_reader.getint(
                "General", "recursion_limit", fallback=50
            ),
            "callbacks": [
   
                CallbackHandler(
                    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        session_id=session_id,
                )
            ],
        }

    @singleton
    @provider
    def provide_proxy(self) -> httpx.Client:
        # return httpx.Client(proxy=os.getenv("HTTP_PROXY_URL"))
        return None

    @singleton
    @provider
    def provide_mongo_client(self) -> MongoClient:
        return MongoClient(
            host=os.getenv("MONGO_HOST"), port=int(os.getenv("MONGO_PORT"))
        )

    @singleton
    @provider
    def provide_nb_executor(self) -> JupyterExecutor:
        server_url = config_reader.get("Jupyter", "server_url")
        token = config_reader.get("Jupyter", "token")
        return JupyterExecutor(server_url, token)

    @singleton
    @provider
    def provide_llm(self, proxy: httpx.Client) -> ChatOpenAI:
        return ChatOpenAI(
            base_url=config_reader.get("API", "base_url"),
            model=config_reader.get("API", "model"),
            http_client=proxy,
            temperature=config_reader.getfloat("API", "temperature"),
        )
    @singleton
    @provider
    def provide_postgres_connection(self) -> Connection:
        conn=Connection.connect(os.getenv("DATABASE_URL"),** {
    "autocommit": True,
    "prepare_threshold": 0,
})
        print("Connected to Postgres")
        return conn.__enter__()
         
    @singleton
    @provider
    def provide_scraper(self, client: MongoClient, config: dict) -> ScrapeKaggle:
        return ScrapeKaggle(client, config)

    @singleton
    @provider
    def provide_planner(
        self, config: dict, proxy: httpx.Client, llm: ChatOpenAI
    ) -> KaggleProblemPlanner:
        return KaggleProblemPlanner(config, proxy, llm=llm)

    @singleton
    @provider
    def provide_re_planner(
        self, config: dict, proxy: httpx.Client, llm: ChatOpenAI
    ) -> KaggleProblemRePlanner:
        return KaggleProblemRePlanner(config, proxy, llm=llm)

    @singleton
    @provider
    def provide_enhancer(
        self, config: dict, proxy: httpx.Client, llm: ChatOpenAI
    ) -> KaggleTaskEnhancer:
        return KaggleTaskEnhancer(config, proxy, llm=llm)

    @singleton
    @provider
    def provide_data_utils(
        self, config: dict, proxy: httpx.Client, llm: ChatOpenAI
    ) -> DataUtils:
        return DataUtils(config, proxy, llm)

    @singleton
    @provider
    def provide_code_agent(
        self, config: dict, proxy: httpx.Client, nb_executor: JupyterExecutor
    ) -> CodeGenerationAgent:
        return CodeGenerationAgent(config, proxy=proxy, nb_executor=nb_executor)

    @singleton
    @provider
    def provide_submission_node(self) -> SubmissionNode:
        return SubmissionNode()

def create_injector():
    app_module = AppModule()
    injector = Injector([app_module])
    return injector, app_module
