# main.py
import argparse
from config_reader import config_reader
from di_container import create_injector, SandboxManager
from agent import KaggleProblemSolver
from persistence.mongo import MongoDBSaver
from dotenv import load_dotenv
from pymongo import MongoClient

def main():
    print(".env loaded:", load_dotenv())

    parser = argparse.ArgumentParser("kaggle_solver")
    parser.add_argument(
        "--url",
        help="url to challenge",
        type=str,
        required=False,
        default=config_reader.get('Kaggle', 'default_challenge_url')
       )
    args = parser.parse_args()

    # Create the injector and get the AppModule
    injector, app_module = create_injector()

    # Use the SandboxManager as a context manager
    with app_module.sandbox_manager as server:
        # Get the MongoDB client from the injector
        mongo_client = injector.get(MongoClient)
        checkpointer = MongoDBSaver(mongo_client, db_name=config_reader.get('MongoDB', 'db_name'))

        # Get the KaggleProblemSolver instance from the injector
        solver = injector.get(KaggleProblemSolver)

        # Compile and invoke the solver
        graph = solver.compile(checkpointer)
        solver.invoke(args.url, debug=True)

    # The SandboxManager context is automatically closed here

if __name__ == "__main__":
    main()