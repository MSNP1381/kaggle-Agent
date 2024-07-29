from langchain.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser


from states.enhancer import EnhancedTask
from states.main import KaggleProblemState


class KaggleTaskEnhancer():
    def __init__(self, config, proxy):
        self.config = config
        self.llm = ChatOpenAI(model="gpt-4o-mini", http_client=proxy, temperature=0)

        self.task_enhancement_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an AI assistant specializing in enhancing and evaluating Kaggle machine learning tasks. Your goal is to enhance task descriptions by combining reasoning and actionable insights.

            **Reasoning**:
            - **Problem Description**: Understand the overall goal and context of the Kaggle problem.
            - **Dataset Information**: Evaluate the dataset’s structure, size, and characteristics.
            - **Current Task**: Define the specifics of the task and its role in the project.
            - **Previous Tasks and Results**: Review what has been done and the outcomes.
            - **Model Information**: Consider existing models and decisions made.
            - **Planned Tasks**: Check alignment with the overall project plan.
            - **Evaluation Metric**: Align the task with the performance metric.
            - **Best Score**: Aim to improve the current best performance.

            **Action**:
            - Enhance the task description to be **specific** and **actionable**.
            - Ensure it **aligns** with the **problem description** and **evaluation metric**.
            - Build upon **previous tasks** and their results.
            - Provide clear **guidance** on what needs to be done and **why**.
            - Specify if **outputs** are required and ensure they are executable by code.

            {format_instructions}
            """,
                ),
                (
                    "human",
                    """
            **Problem Description**: {problem_description}

            **Current Task**: {task}

            **Project State**:
            - Dataset Info: {dataset_info}
            - Tasks Results: {task_results}
            - Model Info: {model_info}
            - Planned Tasks: {planned_tasks}
            - Evaluation Metric: {evaluation_metric}
            - Best Score: {best_score}

            Using the above information, apply reasoning to enhance the task description and determine specific actions required.
            """,
                ),
            ]
        )

    def enhance_task(self, task, state: KaggleProblemState):
        output_parser = PydanticOutputParser(pydantic_object=EnhancedTask)
        format_instructions = output_parser.get_format_instructions()

        response = (self.task_enhancement_prompt | self.llm | output_parser).invoke(
            {
                "task": task,
                "problem_description": state.problem_description,
                "dataset_info": str(state.dataset_info),
                # 'previous_tasks': str(state.previous_tasks),
                "task_results": state.get_task_results(),
                "model_info": str(state.model_info),
                "planned_tasks": str(state.planned_tasks),
                "evaluation_metric": state.evaluation_metric,
                "best_score": state.best_score,
                "format_instructions": format_instructions,
            },
            config=self.config,
        )

        return response

    def __call__(self, state: KaggleProblemState):
        task = state.planned_tasks.pop(0)

        enhanced_task = self.enhance_task(task, state)

        return {"planned_tasks": state.planned_tasks, "enhanced_task": enhanced_task}
