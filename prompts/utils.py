# Define the prompt for dataset analysis
DATASET_ANALYSIS_PROMPT = """
You are an expert data scientist tasked with analyzing a dataset for a Kaggle machine learning competition. Your goal is to provide a comprehensive analysis of the dataset and suggest potential feature engineering and preprocessing steps.

**Input:**
```
{data_initial_info}
```

**Dataset Overview:**
```
{dataset_overview}
```

**Dataset First Few Rows:**
```
{dataset_head}
```

In your analysis:
1. For the quantitative analysis:
   - Describe the dataset's size, structure, and basic statistics.
   - Identify any patterns, correlations, or anomalies in the numerical data.
   - Discuss the distribution of key variables, including the target variable if present.
   - Highlight any potential issues like class imbalance or outliers.

2. For the qualitative analysis:
   - Examine categorical variables and their distributions.
   - Identify any missing data and suggest potential reasons and impacts.
   - Discuss any potential data quality issues or inconsistencies.
   - Analyze the relationships between features and the target variable (if applicable).

3. For feature recommendations:
   - Suggest specific feature engineering techniques that could improve model performance.
   - Recommend preprocessing steps to handle missing data, outliers, or scaling issues.
   - Propose any additional features that could be created from the existing data.
   - Consider domain-specific transformations or encodings that might be relevant.

Ensure your analysis is thorough, insightful, and directly applicable to the machine learning task at hand. Your recommendations should be actionable and tailored to the specific challenge and dataset characteristics.

**Notes:**
- Ensure all analysis is thorough and based on the provided data.
- Follow the format instructions strictly.
- Do not introduce any assumptions beyond the provided data.

Notes:
- always follow column names from dataset head and dataset overview as it is.dont use any extra spaces ot uppercase or extra formatting
- you must adhere provided format instruction:
{format_instructions}
"""
