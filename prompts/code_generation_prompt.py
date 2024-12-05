import random

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

libraries = open("./notebook_requirements.txt").read().split("\n")
random.shuffle(libraries)
pkg_str = ", ".join([f"`{p}`" for p in libraries])


IMPROVED_CODE_GEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
"""
You are a data sciene and machine learning expert. Your task is to generate a code for the given task in a continous flow according to previous generated cpdes and results. 
You are in a ipython and jupyter notebook environment, Generate code for next cell acording to current task provided.
Pay attention to prevoius codes and for new cell continue integrity of code and solution.
in your code generation note that utilization matters for example use n_jobs=-1 in using scikit models and other things

## if data is loaded in previous codes use then and never write redundant data loading use prevoius variables in last generated code
if you Want to load new data use these descriptions:

NOTE: consider memory and resource limitations and write a utilized code 

<LOAD_NEW_DATA>
Data Handling:
- Input directory: "./input/"
    files are:
    1. ./input/train.csv
    2. ./input/overview.md
- Output directory: "./output/"

</LOAD_NEW_DATA>
PROJECT SPECIFICATIONS
---------------------
Problem:
<Problem_DESCRIPTION>
{problem_description}
</Problem_DESCRIPTION>

Evaluation Metric: {evaluation_metric}

** Libraries for code generation,Please dont use other libraries**: {pkg_str}

""",
        ),
        MessagesPlaceholder("history"),
        (
            "user",
            """\
please write code for this task.
Note : ** Please skip visaulization and using plots**

<CurrentTask>

{current_task}

</CurrentTask>
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
## if data is loaded in previous codes use then and never write redundant data loading use prevoius variables in last generated code
in your code generation note that utilization matters for example use n_jobs=-1 in using scikit models and other things

PROJECT SPECIFICATIONS
---------------------
Problem:
'''

{problem_description}

'''


Evaluation Metric: {evaluation_metric}


Available Libraries for code generation: {pkg_str}


""",
        ),
        MessagesPlaceholder("history"),
        (
            "user",
            """
Please fix the error in the code and provide the corrected code.

For using data always Look at the previous cell and its variables and try to use them.

if you Want to load new data use these descriptions:

<LOAD_NEW_DATA>
Data Handling:
- Input directory: "./input/"
    files are:
    1. ./input/train.csv
    2. ./input/overview.md
- Output directory: "./output/"

</LOAD_NEW_DATA>
<CodeWithError>

```{error_code}```

</CodeWithError>

<ErrorMessage>
{error_msg}
</ErrorMessage>


<ErrorResoning>

{suggested_fix}
</ErrorResoning>
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
6. For using data always Look at the previous cell and its variables and try to use them.
7. in your code generation note that utilization matters for example use n_jobs=-1 in using scikit models and other things
## if data is loaded in previous codes use then and never write redundant data loading use prevoius variables in last generated code

if you Want to load new data use these descriptions:
<LOAD_NEW_DATA>
Data Handling:
- Input directory: "./input/"
    files are:
    1. ./input/train.csv
    2. ./input/overview.md
- Output directory: "./output/"

</LOAD_NEW_DATA>



PROJECT SPECIFICATIONS
---------------------
Problem:
'''

{problem_description}

'''


Evaluation Metric: {evaluation_metric}


Available Libraries for code generation: {pkg_str}


Error message from previous attempt: {error_msg}

Please generate a new solution that addresses the current task and avoids previous errors.
""",
        ),
        MessagesPlaceholder("history"),
    ]
)
