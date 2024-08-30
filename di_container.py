# di_container.py
from injector import Injector, Module, singleton, provider
import httpx
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from code_generation_agent import CodeGenerationAgent
from kaggle_scaper import ScrapeKaggle
from nbexecuter_e2b import E2B_executor, SandboxManager
from planner_agent import KaggleProblemPlanner
from replanner import KaggleProblemRePlanner
from task_enhancer import KaggleTaskEnhancer
from dataUtils_agent import KaggleDataUtils
import os
from config_reader import config_reader 
from e2b_code_interpreter import CodeInterpreter
import time

class AppModule(Module):
    
    def __init__(self):
        self.sandbox_manager = SandboxManager()
        self.server = None

    def configure(self, binder):
        binder.bind(SandboxManager, to=self.sandbox_manager)

    @singleton
    @provider
    def provide_config(self) -> dict:
        return {
            "configurable": {"thread_id": str(int(time.time()))},
            "recursion_limit": config_reader.getint('General', 'recursion_limit', fallback=50),
        }

    @singleton
    @provider
    def provide_proxy(self) -> httpx.Client:
        return httpx.Client(proxy=os.getenv("HTTP_PROXY_URL"))

    @singleton
    @provider
    def provide_mongo_client(self) -> MongoClient:
        return MongoClient(
            host=os.getenv("MONGO_HOST"),
            port=int(os.getenv("MONGO_PORT"))
        )

    @singleton
    @provider
    def provide_server(self) -> CodeInterpreter:
        if self.server is None:
            self.server = self.sandbox_manager.__enter__()
        return self.server

    @singleton
    @provider
    def provide_nb_executor(self, server: CodeInterpreter) -> E2B_executor:
        return E2B_executor(server)

    @singleton
    @provider
    def provide_llm(self, proxy: httpx.Client) -> ChatOpenAI:
        return ChatOpenAI(
            base_url=config_reader.get('API', 'base_url'),
            model=config_reader.get('API', 'model'),
            http_client=proxy,
            temperature=config_reader.getfloat('API', 'temperature')
        )

    @singleton
    @provider
    def provide_scraper(self, client: MongoClient, config: dict) -> ScrapeKaggle:
        return ScrapeKaggle(client, config)

    @singleton
    @provider
    def provide_planner(self, config: dict, proxy: httpx.Client, llm: ChatOpenAI) -> KaggleProblemPlanner:
        return KaggleProblemPlanner(config, proxy, llm=llm)

    @singleton
    @provider
    def provide_re_planner(self, config: dict, proxy: httpx.Client, llm: ChatOpenAI) -> KaggleProblemRePlanner:
        return KaggleProblemRePlanner(config, proxy, llm=llm)

    @singleton
    @provider
    def provide_enhancer(self, config: dict, proxy: httpx.Client, llm: ChatOpenAI) -> KaggleTaskEnhancer:
        return KaggleTaskEnhancer(config, proxy, llm=llm)

    @singleton
    @provider
    def provide_data_utils(self, config: dict, proxy: httpx.Client,llm: ChatOpenAI) -> KaggleDataUtils:
        return KaggleDataUtils(config, proxy,llm)

    @singleton
    @provider
    def provide_code_agent(self, config: dict, proxy: httpx.Client, nb_executor: E2B_executor) -> CodeGenerationAgent:
        return CodeGenerationAgent(config, proxy=proxy, nb_executor=nb_executor)
    
def create_injector():
    app_module = AppModule()
    injector = Injector([app_module])
    return injector, app_module