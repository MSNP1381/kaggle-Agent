import kaggle
from states.main import KaggleProblemState

class SubmissionNode():

    def __call__(self, state: KaggleProblemState, config: dict):
        competition = config.get("competition")
        submission_file = config.get("submission_file")
        message = config.get("submission_message", "Automated submission")

        if not competition or not submission_file:
            raise ValueError("Competition name and submission file must be provided.")

        # Submit the file to the competition
        print(f"Submitting {submission_file} to {competition} with message: {message}")
        kaggle.api.competition_submit(submission_file, message, competition)

        # Retrieve the submission result
        submissions = kaggle.api.competition_submissions(competition)
        latest_submission = max(submissions, key=lambda x: x['date'])
        print(f"Latest submission result: {latest_submission}")

        # Update the state with the latest submission result
        state.latest_submission = latest_submission
        return state