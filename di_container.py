# di_container.py
from injector import Injector, Module, singleton, provider
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from code_manager import KaggleCodeManager
from code_generation_agent import CodeGenerationAgent
from kaggle_scraper import ScrapeKaggle
from psycopg import Connection

# from nbexecuter_e2b import E2B_executor, SandboxManager
from executors.nbexecutor_jupyter import JupyterExecutor

# from replanner import KaggleProblemRePlanner
from planner_agent import KaggleProblemPlanner
from task_enhancer import KaggleTaskEnhancer
from data_utils import DataUtils
import os
from config_reader import config_reader

from langfuse.callback import CallbackHandler
import time

from submission.submission import SubmissionNode  # Correct Python import
from states.memory import MemoryAgent  # Add this import


class AppModule(Module):
    def __init__(self):
        self.server = None

    @singleton
    @provider
    def provide_config(self) -> dict:
        return {
            "configurable": {"thread_id": str(int(time.time()))},
            "recursion_limit": config_reader.getint(
                "General", "recursion_limit", fallback=50
            ),
            "callbacks": [CallbackHandler(session_id=f"session-{int(time.time())}")],
        }

    @singleton
    @provider
    def provide_mongo_client(self) -> MongoClient:
        mongo_uri = f"mongodb://{os.getenv('MONGO_HOST')}:{os.getenv('MONGO_PORT')}"
        return MongoClient(mongo_uri)

    @singleton
    @provider
    def provide_nb_executor(self) -> JupyterExecutor:
        server_url = config_reader.get("Jupyter", "server_url")
        token = config_reader.get("Jupyter", "token")
        return JupyterExecutor(server_url, token)

    @singleton
    @provider
    def provide_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            base_url=config_reader.get("API", "base_url"),
            model=config_reader.get("API", "model"),
            temperature=config_reader.getfloat("API", "temperature"),
        )

    @singleton
    @provider
    def provide_postgres_connection(self) -> Connection:
        conn = Connection.connect(
            os.getenv("DATABASE_URL"),
            **{
                "autocommit": True,
                "prepare_threshold": 0,
            },
        )
        print("Connected to Postgres")
        return conn.__enter__()

    @singleton
    @provider
    def provide_scraper(self, client: MongoClient, config: dict) -> ScrapeKaggle:
        return ScrapeKaggle(client, config)

    @singleton
    @provider
    def provide_planner(
        self, config: dict, llm: ChatOpenAI, memory: MemoryAgent
    ) -> KaggleProblemPlanner:
        return KaggleProblemPlanner(config, llm=llm, memory=memory)

    # @singleton
    # @provider
    # def provide_re_planner(
    #     self, config: dict, proxy: httpx.Client, llm: ChatOpenAI
    # ) -> KaggleProblemRePlanner:
    #     return KaggleProblemRePlanner(config, proxy, llm=llm)

    @singleton
    @provider
    def provide_enhancer(
        self, config: dict, llm: ChatOpenAI, memory_agent: MemoryAgent
    ) -> KaggleTaskEnhancer:
        return KaggleTaskEnhancer(config, llm, memory_agent)

    @singleton
    @provider
    def provide_memory_agent(self, llm: ChatOpenAI, mongo: MongoClient) -> MemoryAgent:
        return MemoryAgent(llm=llm, mongo=mongo)

    @singleton
    @provider
    def provide_data_utils(
        self, config: dict, llm: ChatOpenAI, mongo_client: MongoClient
    ) -> DataUtils:
        return DataUtils(config, llm, mongo_client)

    @singleton
    @provider
    def provide_code_agent(
        self,
        config: dict,
        llm: ChatOpenAI,
        nb_executor: JupyterExecutor,
        memory_agent: MemoryAgent,
    ) -> CodeGenerationAgent:
        return CodeGenerationAgent(
            llm, config, nb_executor=nb_executor, memory_agent=memory_agent
        )

    @singleton
    @provider
    def provide_submission_node(self) -> SubmissionNode:
        return SubmissionNode()

    @singleton
    @provider
    def provide_code_manager(
        self, code_gen_agent: CodeGenerationAgent
    ) -> KaggleCodeManager:
        code_agent = KaggleCodeManager(code_gen_agent)
        return code_agent


def create_injector():
    app_module = AppModule()
    injector = Injector([app_module])
    return injector, app_module
