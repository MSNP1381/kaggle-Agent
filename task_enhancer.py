import httpx
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

class EnhancedTask(BaseModel):
    task: str = Field(description="The original task description")
    enhanced_description: str = Field(description="An enhanced and more detailed description of the task")
    requires_code_output: bool = Field(description="Whether this task requires code execution and output")
    expected_output_type: Optional[str] = Field(description="The expected type of output if code execution is required (e.g., 'dataframe', 'plot', 'metric', 'model')")

class KaggleTaskEnhancer:
    def __init__(self, config):
        self.config = config
        proxy=httpx.Client(proxy="http://127.0.0.1:2081")
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", http_client=proxy)

        self.task_enhancement_prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant tasked with enhancing and evaluating Kaggle machine learning tasks.
        Given a task from the problem-solving plan, provide a more detailed description and determine if code execution is required.

        Original Task: {task}

        Problem Description:
        {problem_description}

        Project State:
        {state}

        Your task is to:
        1. Provide a more detailed and specific description of the task.
        2. Determine if this task requires code execution and output.
        3. If code execution is needed, specify the expected type of output.

        {format_instructions}
        """)

    def enhance_task(self, task, state):
        output_parser = PydanticOutputParser(pydantic_object=EnhancedTask)
        format_instructions = output_parser.get_format_instructions()
        
        response = (self.task_enhancement_prompt | self.llm | output_parser).invoke({
            'task': task,
            'problem_description': state.problem_description,
            'state': str(state.__dict__),
            'format_instructions': format_instructions
        }, config=self.config)
        
        return response
