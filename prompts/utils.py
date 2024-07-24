# Define the prompt for dataset analysis
DATASET_ANALYSIS_PROMPT = """
You are an AI assistant tasked with analyzing a dataset for a Kaggle machine learning problem.
Given the dataset, provide a comprehensive description and recommend preprocessing steps.

Dataset Overview:
{dataset_overview}

Your task is to:
1. Describe the dataset including data types, number of missing values, unique values, and any potential issues.
2. Recommend preprocessing steps to prepare the data for machine learning.

Respond with a detailed description and a list of preprocessing steps.
"""