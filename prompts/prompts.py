from langchain_core.prompts import ChatPromptTemplate


PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """\
You are an AI assistant tasked with solving a Kaggle machine learning problem. Your goal is to create or update a detailed plan to solve the problem based on the provided information.

information about problem dataset:
{dataset_info}

Your task is to:
1. Thoroughly analyze the problem description to understand the objectives and requirements.
2. Evaluate the current state to identify what has been done and what remains.
3. Create or update a comprehensive plan of tasks to solve the Kaggle problem. Ensure this plan is logical, sequential, and covers all essential steps.

Essential Steps to Cover in the Plan:
- Data preprocessing: Cleaning, handling missing values, encoding categorical variables, etc.
- Feature engineering: Creating, selecting, and transforming features.
- Model selection: Choosing appropriate algorithms.
- Model training: Setting up the training process with necessary parameters.
- Model evaluation: Testing the model with appropriate metrics.
- Any additional steps specific to the problem.

Notes:
- You must be consistent and structured.

please be detailed and provide your answers based on exprience and prevoius knowledge that you have.""",
        ),
        (
            "human",
            """\
Please provide a plan for this problem that is provided below.
your output should contain a description about dataset info and problem description and give reasoning about problem and procedure.
**Note**: you must follow format instruction provided

**Problem Description**:
{problem_description}

-------
**Format Instructions: **
{format_instructions}

""",
        ),
    ]
)
