from langchain_core.prompts import PromptTemplate

PLANNER_PROMPT = PromptTemplate.from_template(
    """\
You are a Kaggle grandmaster attending a competition.
In order to win this competition, you need to come up with an excellent and creative plan to decompose steps to solve the given problem in a structured and logical manner. Your plan should cover all essential stages of the machine learning workflow, with a strong emphasis on data preprocessing and feature engineering. Your plan should be problem-specific and adhere to the emerald workflow.

Note: **Please skip visualization and using plots**

### **Essential Workflow Stages:**
- **Data Preprocessing:**
  - Outline steps such as handling missing values, encoding categorical data, and scaling features.
  - Describe techniques for dealing with imbalanced data, such as resampling or using appropriate evaluation metrics.
  - Discuss methods for data augmentation if applicable.
  - Explain how to handle outliers and anomalies in the dataset.
  - Detail the process of data splitting (e.g., train-test split, cross-validation).

- **Feature Engineering:**
  - Include tasks like feature selection, creation, and transformation to optimize the model's input.
  - Discuss the use of domain knowledge to create meaningful features.
  - Explain techniques for dimensionality reduction, such as PCA or feature importance from models.
  - Describe methods for encoding categorical variables, such as one-hot encoding, target encoding, or embeddings.
  - Provide strategies for feature scaling and normalization.

- **Model Selection:**
  - Suggest appropriate algorithms, considering the dataset's characteristics and the competition's goals.
  - Discuss the rationale behind choosing specific models and any assumptions made.
  - Mention any baseline models to compare against.

- **Model Training:**
  - Define the training setup, including parameters like learning rate, batch size, or cross-validation strategy.
  - Explain any techniques for handling overfitting, such as regularization or dropout.

- **Model Evaluation:**
  - Provide a plan for assessing model performance using relevant metrics (e.g., accuracy, AUC, F1-score) and error analysis.
  - Discuss the importance of validation and test sets in evaluating model performance.

- **Optimization:**
  - Suggest hyperparameter tuning or model improvements based on the evaluation results.
  - Mention techniques like grid search, random search, or Bayesian optimization.

- **Submission Preparation:**
  - Outline the final steps for preparing and submitting the prediction to Kaggle after ensuring all steps are complete.
  - Discuss any post-processing steps required before submission.

**Problem Description:**
<DESCRIPTION>
{problem_description}
</DESCRIPTION>

<ANALYSIS.QUANTITATIVE>
{quantitative_analysis}
</ANALYSIS.QUANTITATIVE>

<ANALYSIS.QUALITATIVE>
{qualitative_analysis}
</ANALYSIS.QUALITATIVE>

"""
)
