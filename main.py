# main.py

import logging

import coloredlogs
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
import time
from agent import KaggleProblemSolver
from config_reader import config_reader
from di_container import create_injector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("kaggle_agent.log"),
        logging.StreamHandler(),
    ],
)

coloredlogs.install()
logger = logging.getLogger(__name__)


def main():
    print(".env loaded:", load_dotenv(override=True))

    url = config_reader.get("Kaggle", "default_challenge_url")
    # thread_Id = config_reader.get("thread", "thread_id",str(int(time.time())))
    thread_Id = str(int(time.time()))

    # Create the injector and get the AppModule
    injector, app_module = create_injector()

    postgres_client = injector.get(Connection)

    checkpointer = PostgresSaver(postgres_client)

    # Get the KaggleProblemSolver instance from the injector
    solver = injector.get(KaggleProblemSolver)

    # Compile and invoke the solver
    solver.compile(None)
    if not thread_Id:
        thread_Id = str(int(time.time() * 1000))
    print(
        "*" * 20,
        "\n\n\n",
        f"Thread ID: {thread_Id}",
        "\n\n\n",
        "*" * 20,
    )
    solver.invoke(url, thread_Id)
    checkpointer.close()


if __name__ == "__main__":
    main()
