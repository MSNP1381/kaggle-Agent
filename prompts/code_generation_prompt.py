import random

from langchain.prompts import PromptTemplate, ChatPromptTemplate

libraries = open("./notebook_requirements.txt").read().split("\n")
random.shuffle(libraries)
pkg_str = ", ".join([f"`{p}`" for p in libraries])


IMPROVED_CODE_GEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a python code generation expert in machine learning and data science. Your task is to generate a code for the given task
You are in a notebook  environment, Generate code for next notebook cell acording to current task provided.
Pay attention to prevoius codes and for new cell continue integrity of code and solution.

PROJECT SPECIFICATIONS
---------------------
Problem:
'''

{problem_description}

'''


Evaluation Metric: {evaluation_metric}


Available Libraries for code generation: {pkg_str}

TECHNICAL REQUIREMENTS
---------------------
Data Handling:
- Input directory: "./input/"
    files are:
    1. ./input/train.csv
    2. ./input/test.csv
    3. ./input/sample_submission.csv
    4. ./input/overview.md
- Output directory: "./output/"
""",
        ),
        ("placeholder", "{history}"),
        (
            "user",
            """\
please write code for this task.

Current Task: '''

{current_task}

'''

""",
        ),
    ]
).partial(pkg_str=pkg_str)


DEBUGGING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
you are a python code debugger debug cpde based on previous code and error message provided.

""",
        ),
        ("placeholder", "{history}"),
        (
            "user",
            """
Please fix the error in the code and provide the corrected code.

Current code:

```{current_code}```



resoning on error:

{resoning}
            """,
        ),
    ]
)

NEW_SOLUTION_PROMPT = PromptTemplate.from_template(
    """
You are a Kaggle grandmaster expert in machine learning and data science. Your task is to generate a new solution for the given problem, taking into account previous attempts and errors.

Guidelines:
1. Analyze the previous code and error messages.
2. Propose a new approach that avoids the previous errors.
3. Implement the new solution as a complete, self-contained Python script.
4. Explain your reasoning for the new approach.
5. Ensure the new solution addresses the current task and overall problem goal.

Problem description: {problem_description}
Current task: {current_task}
Evaluation metric: {evaluation_metric}
Previous task results: {previous_tasks}
Error message from previous attempt: {error_msg}

Please generate a new solution that addresses the current task and avoids previous errors.
"""
)
