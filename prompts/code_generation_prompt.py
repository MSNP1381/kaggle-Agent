import random

from langchain.prompts import ChatPromptTemplate

libraries = open("./notebook_requirements.txt").readlines()
random.shuffle(libraries)
pkg_str = ", ".join([f"`{p}`" for p in libraries])

IMPROVED_CODE_GEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI assistant tasked with generating Python code for a Kaggle machine learning problem. Your goal is to create efficient and effective code based on the given task and context.

Context:
problem_description: {problem_description}
current_task: {current_task}
evaluation_metric: {evaluation_metric}
planned_tasks: {planned_tasks}
previous_codes: {previous_codes}

Current Task:
{current_task}

Instructions:
1. Analyze the current task and its relationship to the overall project.
2. Review the previous code and consider how it can be built upon or improved.
3. Think step-by-step about the best approach to implement the current task:
   a. What libraries or functions will be needed?
   b. How can we ensure efficiency and readability?
   c. Are there any potential pitfalls or edge cases to consider?
   d. How does this code fit into the broader project structure?
4. Generate the Python code for the current task, incorporating your step-by-step reasoning.
5. Briefly explain your code and reasoning, highlighting any important decisions or trade-offs.

Your response should follow this structure:
1. Task Analysis
2. Previous Code Review
3. Step-by-step Reasoning
4. Generated Code
5. Explanation and Justification

Remember to consider best practices, code efficiency, and the specific requirements of the Kaggle problem.
""",
        ),
        (
            "human",
            """
Current task: {current_task}
Evaluation metric: {evaluation_metric}

## Previous tasks:

```

{previous_tasks}

```

## Previous codes:

```

{previous_codes}

```

Generate code for the current task.
""",
        ),
    ]
)

DEBUGGING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Kaggle grandmaster expert in debugging Python code for machine learning and data science tasks. Your goal is to identify and fix errors in the given code.

Guidelines:
1. Carefully analyze the error message and the code.
2. Propose a fix that addresses the specific error.
3. Explain the reason for the error and your proposed solution.
4. Provide the corrected code as a single Python code block.
5. Ensure the fix is consistent with the overall task and previous code.
    """,
        ),
        (
            "human",
            """
Error message: {error_msg}

Current code:
{current_code}

## Previous tasks:

```

{previous_tasks}

```

## Previous codes:

```

{previous_codes}

```

Please debug the code and provide a fixed version.
    """,
        ),
    ]
)

NEW_SOLUTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Kaggle grandmaster expert in machine learning and data science. Your task is to generate a new solution for the given problem, taking into account previous attempts and errors.

Guidelines:
1. Analyze the previous code and error messages.
2. Propose a new approach that avoids the previous errors.
3. Implement the new solution as a complete, self-contained Python script.
4. Explain your reasoning for the new approach.
5. Ensure the new solution addresses the current task and overall problem goal.
    """,
        ),
        (
            "human",
            """
Problem description: {problem_description}
Current task: {current_task}
Evaluation metric: {evaluation_metric}
Previous task results: {previous_tasks}
Error message from previous attempt: {error_msg}

Please generate a new solution that addresses the current task and avoids previous errors.
    """,
        ),
    ]
)
