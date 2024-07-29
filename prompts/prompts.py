from langchain_core.prompts import ChatPromptTemplate


PLANNER_PROMPT = """
You are an AI assistant tasked with solving a Kaggle machine learning problem. Your goal is to create or update a detailed plan to solve the problem based on the provided information.

Problem Description:
{problem_description}

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
- Do not call the plt.show() method for plots in the code.
- Do not use any visually formatted outputs like rendered HTML; use text or dictionaries for outputs.
- Take into account the current state, including results from previous tasks and existing variables.
- You must be consistent and structured.


formated instruction : is as follow
{format_instructions}

Provide sufficient detail in each task to guide subsequent steps.
"""
