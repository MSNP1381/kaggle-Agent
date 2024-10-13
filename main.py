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
    print(".env loaded:", load_dotenv(override=True))

    url = config_reader.get("Kaggle", "default_challenge_url")

    # Create the injector and get the AppModule
    injector, app_module = create_injector()

    postgres_client = injector.get(Connection)

    checkpointer = PostgresSaver(postgres_client)

    # Get the KaggleProblemSolver instance from the injector
    solver = injector.get(KaggleProblemSolver)

    # Compile and invoke the solver
    graph = solver.compile(checkpointer)
    solver.invoke(url)


if __name__ == "__main__":
    main()
