from langchain.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

from states.enhancer import EnhancedTask
from states.main import KaggleProblemState


class KaggleTaskEnhancer:
    def __init__(self, config, proxy):
        self.config = config
        self.llm = ChatOpenAI(model="gpt-4o-mini", http_client=proxy, temperature=0)

        self.task_enhancement_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an AI assistant specialized in enhancing and evaluating Kaggle machine learning tasks. 
            Your goal is to provide detailed, actionable task descriptions that align with the current state of the project.

            When analyzing a task, consider the following aspects of the project state:
            1. Problem Description: The overall goal and context of the Kaggle problem.
            2. Dataset Information: The structure, size, and characteristics of the dataset.
            3. Current Task: The specific task at hand and its place in the overall workflow.
            4. Previous Tasks: What has been done so far and how it impacts the current task.
            5. Task Results: Outcomes of previously completed tasks.
            6. Model Information: Any existing models or modeling decisions made.
            7. Planned Tasks: The overall plan and how this task fits into it.
            8. Evaluation Metric: The metric used to assess model performance.
            9. Best Score: The current best performance achieved.

            Your enhanced task description should:
            - Be specific and actionable
            - Align with the problem description and evaluation metric
            - Build upon previous tasks and their results
            - Consider the current state of the project and dataset
            - Provide clear guidance on what needs to be done and why
            - Provide wethere outputs are reqired for this task or not which can be executed by code   

            {format_instructions}
            """,
                ),
                (
                    "human",
                    """
            Problem Description: {problem_description}

            Current Task: {task}

            Project State:
            - Dataset I-nfo: {dataset_info}
            - Tasks Results: {task_results}
            - Model Info: {model_info}
            - Planned Tasks: {planned_tasks}
            - Evaluation Metric: {evaluation_metric}
            - Best Score: {best_score}

            Based on this information, please enhance the task description and determine its requirements.
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
