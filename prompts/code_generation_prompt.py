import random
from langchain.prompts import ChatPromptTemplate
libraries=open('./notebook_requirements.txt').readlines()
random.shuffle(libraries)
pkg_str = ", ".join([f"`{p}`" for p in libraries])

IMPROVED_CODE_GEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Kaggle grandmaster expert in machine learning and data science. Generate executable Python code for the given task in a Jupyter notebook environment.

Guidelines:
1. Continue from previous code, avoiding repetition of operations.
2. Implement the proposed solution and print the evaluation metric on a hold-out validation set.
3. Create a self-contained, single-file Python program.
4. Provide a complete script without skipping parts.
5. Use a single code block in your response.
6. Input data is in "./input"; save test predictions as "submission.csv" in "./working".
7. Temporary files can be stored in "./working".
8. Prioritize code efficiency and follow PEP 8 style guidelines.
9. Include error handling and input validation where appropriate.
10. Use this Evaluation Metric: {evaluation_metric}

Context:
- Problem Goal: {problem_description}

Available packages: {pkg_str}
Note: All packages are pre-installed. Prefer PyTorch for neural networks.
    """),
        (
            "human",
            """
Generate new code continuing from:
{previous_task_results}

Your code should:
1. Build upon previous operations and variable definitions.
2. Address the current task: {current_task}
3. Incorporate this context: {relevant_context}
4. Include only new operations, avoiding repetition.
5. Be executable and advance the overall solution.
6. Include brief comments explaining key steps.

Provide your solution as a single Python code block.
    """)
    ]
)

DEBUGGING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are a Kaggle grandmaster expert in debugging Python code for machine learning and data science tasks. Your goal is to identify and fix errors in the given code.

Guidelines:
1. Carefully analyze the error message and the code.
2. Propose a fix that addresses the specific error.
3. Explain the reason for the error and your proposed solution.
4. Provide the corrected code as a single Python code block.
5. Ensure the fix is consistent with the overall task and previous code.
    """),
    ("human", """
Error message: {error_msg}

Current code:
{current_code}

Previous task results:
{previous_task_results}

Please debug the code and provide a fixed version.
    """)
])

NEW_SOLUTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are a Kaggle grandmaster expert in machine learning and data science. Your task is to generate a new solution for the given problem, taking into account previous attempts and errors.

Guidelines:
1. Analyze the previous code and error messages.
2. Propose a new approach that avoids the previous errors.
3. Implement the new solution as a complete, self-contained Python script.
4. Explain your reasoning for the new approach.
5. Ensure the new solution addresses the current task and overall problem goal.
    """),
    ("human", """
Problem description: {problem_description}
Current task: {current_task}
Evaluation metric: {evaluation_metric}
Previous task results: {previous_task_results}
Error message from previous attempt: {error_msg}

Please generate a new solution that addresses the current task and avoids previous errors.
    """)
])
