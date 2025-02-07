from langchain_core.prompts import PromptTemplate

PLANNER_PROMPT = PromptTemplate.from_template(
    """\
You are a Kaggle grandmaster attending a competition.
In order to win this competition, you need to come up with an excellent and creative plan to decompose steps to solve the given problem in a structured and logical manner.
Your plan should cover all essential stages of the machine learning workflow, with a strong emphasis on data preprocessing and feature engineering.
Your plan should be problem-specific and adhere to the emerald workflow problem details are provided below.

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
