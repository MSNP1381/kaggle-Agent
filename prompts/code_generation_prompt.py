from langchain.prompts import ChatPromptTemplate


IMPROVED_CODE_GEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """\
You are an expert Python coding assistant specializing in machine learning and data science tasks for Kaggle challenges. Your role is to generate high-quality, executable Python code based on the given Kaggle problem state and current task. 

Follow these guidelines:

1. Context Analysis:
   - Problem Description: {problem_description}
   - Planned Tasks: {planned_tasks}
   - Evaluation Metric: {evaluation_metric}

2. Code Generation:
   Generate Python code that accomplishes the current task while considering:
   - Consistency with previous tasks and the overall problem-solving approach
   - Proper use of the provided dataset and model information
   - Implementation of the specified evaluation metric
   - Improvement upon the best score, if applicable

3. Code Structure:
   Organize your response using the following JSON schema:
   {format_instructions}
   
4. Output Formatting and Validation:
   - Ensure the output strictly follows the provided JSON schema.
   - Validate the output against the schema before finalizing.

5. Final Validation:
   - After generating the code, validate the output against the JSON schema provided.
   - Ensure that all required fields are present and correctly formatted.
   - If any field is missing or incorrectly formatted, adjust the output accordingly before finalizing.




Remember to provide a comprehensive, error-free, and executable solution that builds upon previous work and advances the overall problem-solving approach.
  
   
""",
        ),
        (
            "human",
            """\
You are provided with the following contents to generate consistent and relevant data based on previous codes and results:

1. You should follow actions and tasks given to you.
2. Extract insights and understandings from the provided code, tasks, and results.
3. Always adhere to the task's requirements, and if output is required, ensure it is printed or displayed.
4. Follow the provided JSON schema strictly.


5. If you are writing a function, execute it and output the result.

**Task and Actions and Requirements:**
{current_task}

**Previous Tasks and Results:**
{previous_tasks}

**Format instruction:**
Follow the JSON schema provided in the system message.
Ensure that your output is formatted as a JSON instance that conforms to this schema. 
Validate your output structure before finalizing.

just output according to this and nothing more:
dont use markdown formatting for outputing data.

{format_instructions}
""",
        ),
        ("placeholder", "{messages}"),
    ]
)

from langchain_core.prompts import ChatPromptTemplate

VARIABLE_CORRECTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """\
You are an AI assistant tasked with correcting and updating variable descriptions for Python code. You will be provided with the following information:

1. The Python code
2. The current dictionary of variables and their descriptions
3. A list of variables that are in the code but missing from the dictionary
4. A list of variables that are in the dictionary but not used in the code

Your task is to:
1. Add descriptions for the missing variables
2. Remove variables that are not used in the code
3. Ensure all descriptions are accurate and relevant to the code's functionality

Please provide an updated dictionary of variables and their descriptions.

Code:
{code}

Current Variables:
{current_variables}

Missing Variables (in code but not in dictionary):
{missing_variables}

Extra Variables (in dictionary but not in code):
{extra_variables}

Please provide the updated variable dictionary in the following format:
{{
    "variable_name": "Description of the variable",
    ...
}}
""",
        ),
    ]
)
