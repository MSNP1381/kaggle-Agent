import kaggle
from langchain_core.tools import tool

submit_to_kaggle_function_description = {
    "name": "submit_to_kaggle",
    "description": "Submit a file to a Kaggle competition and get the submission result",
    "parameters": {
        "type": "object",
        "properties": {
            "competition": {
                "type": "string",
                "description": "The name of the Kaggle competition",
            },
            "submission_file": {
                "type": "string",
                "description": "The path to the file to be submitted",
            },
            "submission_message": {
                "type": "string",
                "description": "The submission message",
                "default": "Automated submission",
            },
        },
        "required": ["competition", "submission_file"],
    },
}


@tool
def submit_to_kaggle(
    competition: str,
    submission_file: str,
    submission_message: str = "Automated submission",
) -> dict:
    """
    Submit a file to a Kaggle competition and return the submission result.

    Args:
    competition (str): The name of the Kaggle competition.
    submission_file (str): The path to the file to be submitted.
    submission_message (str, optional): The submission message. Defaults to "Automated submission".

    Returns:
    dict: The latest submission result.
    """
    if not competition or not submission_file:
        raise ValueError("Competition name and submission file must be provided.")

    print(
        f"Submitting {submission_file} to {competition} with message: {submission_message}"
    )
    kaggle.api.competition_submit(submission_file, submission_message, competition)

    submissions = kaggle.api.competition_submissions(competition)
    latest_submission = max(submissions, key=lambda x: x["date"])
    print(f"Latest submission result: {latest_submission}")

    return latest_submission
