from langchain.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

class EnhancedTask(BaseModel):
    task: str = Field(description="The original task description", default="No Task provided")
    enhanced_description: str = Field(description="An enhanced and more detailed description of the task", default="no enhancement")
    requires_code_output: bool = Field(description="Whether this task requires code execution and output", default=False)
    expected_output_type: str = Field(description="The expected type of output if code execution is required (e.g., 'dataframe', 'plot', 'metric', 'model')", default="No Expected Output")
    dependencies: list = Field(description="List of tasks that this task depends on", default_factory=list)
    estimated_time: str = Field(description="Estimated time to complete this task", default="Unknown")

class KaggleTaskEnhancer:
    def __init__(self, config, proxy):
        self.config = config
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", http_client=proxy, temperature=0)

        self.task_enhancement_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant specialized in enhancing and evaluating Kaggle machine learning tasks. 
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

            {format_instructions}
            """),
            ("human", """
            Problem Description: {problem_description}

            Current Task: {task}

            Project State:
            - Dataset Info: {dataset_info}
            - Previous Tasks: {previous_tasks}
            - Task Results: {task_results}
            - Model Info: {model_info}
            - Planned Tasks: {planned_tasks}
            - Evaluation Metric: {evaluation_metric}
            - Best Score: {best_score}

            Based on this information, please enhance the task description and determine its requirements.
            """)
        ])

    def enhance_task(self, task, state):
        output_parser = PydanticOutputParser(pydantic_object=EnhancedTask)
        format_instructions = output_parser.get_format_instructions()
        
        response = (self.task_enhancement_prompt | self.llm).invoke({
            'task': task,
            'problem_description': state.problem_description,
            'dataset_info': str(state.dataset_info),
            'previous_tasks': str(state.previous_tasks),
            'task_results': str(state.task_results),
            'model_info': str(state.model_info),
            'planned_tasks': str(state.planned_tasks),
            'evaluation_metric': state.evaluation_metric,
            'best_score': state.best_score,
            'format_instructions': format_instructions
        }, config=self.config)
        
        return output_parser.invoke(response)