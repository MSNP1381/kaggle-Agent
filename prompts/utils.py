# Define the prompt for dataset analysis
DATASET_ANALYSIS_PROMPT = """
You are an AI assistant tasked with analyzing a dataset for a Kaggle machine learning problem.
Given the dataset, provide a comprehensive description and recommend preprocessing steps.

Dataset Overview:
{dataset_overview}

Dataset first few rows:
```
{dataset_head}
```
Your task is to:
1. Describe the dataset including data types, number of missing values, unique values, and any potential issues in manner of qualitative output.
    for example:
    ```
    Columns: ['Home', 'Price', 'SqFt', 'Bedrooms', 'Bathrooms', 'Offers', 'Brick', 'Neighborhood']
    Data Types:
    Home             discrete
    Price            discrete
    SqFt             discrete
    Bedrooms         discrete
    Bathrooms        discrete
    Offers           discrete
    Brick           categorical
    Neighborhood    categorical
    Missing Values:
    Home            None
    Price           None
    SqFt            None
    Bedrooms        None
    Bathrooms       None
    Offers          None
    Brick           None
    Neighborhood    None
    dtype: int64
    Unique Values:
    Home            128
    Price           123
    SqFt             61
    Bedrooms          4
    Bathrooms         3
    Offers            6
    Brick             2
    Neighborhood      3
    dtype: int64""
    ```
2. based on provided columns and data types please provide a summarized information about what to do and feature selection and even feature selection based on column names 

Notes:
- always follow column names from dataset head and dataset overview as it is.dont use any extra spaces ot uppercase or extra formatting 
- you must adhere provided format instruction:
{format_instructions}
"""
