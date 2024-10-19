# main.py

from psycopg import Connection
from config_reader import config_reader
from di_container import create_injector
from agent import KaggleProblemSolver
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
import coloredlogs
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
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

    # Create the injector and get the AppModule
    injector, app_module = create_injector()

    postgres_client = injector.get(Connection)

    checkpointer = PostgresSaver(postgres_client)

    # Get the KaggleProblemSolver instance from the injector
    solver = injector.get(KaggleProblemSolver)

    # Compile and invoke the solver
    solver.compile(checkpointer)
    solver.invoke(url)


if __name__ == "__main__":
    main()
