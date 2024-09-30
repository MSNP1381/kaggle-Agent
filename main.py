# main.py
import argparse
import time

from psycopg import Connection
from config_reader import config_reader
from di_container import create_injector
from agent import KaggleProblemSolver
from persistence.mongo import MongoDBSaver
from dotenv import load_dotenv
from pymongo import MongoClient
from langgraph.checkpoint.postgres import PostgresSaver

def main():
    print(".env loaded:", load_dotenv())

    url = config_reader.get("Kaggle", "default_challenge_url")

    # Create the injector and get the AppModule
    injector, app_module = create_injector()

    # Use the SandboxManager as a context manager

    # Get the MongoDB client from the injector
    # mongo_client = injector.get(MongoClient)
    postgres_client = injector.get(Connection)
    
    # checkpointer = MongoDBSaver(
    #     mongo_client, db_name=config_reader.get("MongoDB", "db_name")
    # )
    
    checkpointer = PostgresSaver(postgres_client)
    checkpointer.setup()
    # checkpointer.setup()
    # Get the KaggleProblemSolver instance from the injector
    solver = injector.get(KaggleProblemSolver)

    # Compile and invoke the solver
    graph = solver.compile(checkpointer)
    # return graph
    solver.invoke(url)

    # The SandboxManager context is automatically closed here


if __name__ == "__main__":
    main()
