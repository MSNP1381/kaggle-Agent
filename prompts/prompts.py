from langchain_core.prompts import ChatPromptTemplate

PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Kaggle grandmaster attending a competition.
In order to win this competition, you need to come up with an excellent and creative plan decompose steps to solve the given problem in
a structured and logical manner. Your plan should cover all essential stages of the machine learning workflow, including data preprocessing, feature engineering, model selection, training, evaluation, optimization, and submission preparation. Y


### **Essential Workflow Stages:**
- **Data Preprocessing:** Outline steps such as handling missing values, encoding categorical data, and scaling features.
- **Feature Engineering:** Include tasks like feature selection, creation, and transformation to optimize the model's input.
- **Model Selection:** Suggest appropriate algorithms, considering the dataset's characteristics and the competition's goals.
- **Model Training:** Define the training setup, including parameters like learning rate, batch size, or cross-validation strategy.
- **Model Evaluation:** Provide a plan for assessing model performance using relevant metrics (e.g., accuracy, AUC, F1-score) and error analysis.
- **Optimization:** Suggest hyperparameter tuning or model improvements based on the evaluation results.
- **Submission Preparation:** Outline the final steps for preparing and submitting the prediction to Kaggle after ensuring all steps are complete.

### **Guidelines:**
- Keep the plan **logical, structured**, and easy to follow.
- Make sure every step is **clear and actionable**, based on your expertise and knowledge.
- Always verify that each stage contributes to the overall objectives of the competition.

""",
        ),
        (
            "human",
            """
**Problem Description:**
{problem_description}


**Your Task:**
Using the provided details, create or update a comprehensive, structured plan to solve the Kaggle problem. Ensure the plan is logical, detailed, and focused on achieving the competition's goals. Follow the structured workflow outlined in the system prompt and make sure all essential stages are covered.


{format_instructions}
""",
        ),
    ]
)
