from langchain.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

from states.enhancer import EnhancedTask
from states.main import KaggleProblemState


class KaggleTaskEnhancer:
    def __init__(self, config, proxy, llm: ChatOpenAI):
        self.config = config
        self.llm = llm
        self.task_enhancement_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """\
You are an AI assistant specializing in enhancing tasks and interpreting codes of Kaggle machine learning problems. Your goal is to enhance tasks by combining reasoning and actionable insights.
**Important Notes:**
    1. Always provide a summary of previous codes and tasks to create a consistent code flow for the ML notebook.
    2. Provide your understanding from the results of previous codes and what you have understood from those.
    3. Ensure the generated task is understandable for a code generation LLM Agent. It should be structured for Python and ML tasks for better understanding by the code agent.
    4. Always provide requirements for the task if needed and what is required to get the best result.

**Use the Following Structure:**
    Task: The input task you must answer.

    Thought: You should always think about what to do.

    Actions: The list of actions to take, based on the provided context.

    Observation: The result of the action.

    ... (this Thought/Action/Action Input/Observation can repeat N times)

    Thought: I now know the final answer.

    Final Answer: The final enhanced task to the original input task.
""",
                ),
                (
                    "human",
                    """\
**Task**: {current_task}
**Context**:
            
**Problem Description**: {problem_description}

**Project State**:
    - Dataset Info: {dataset_info}
    - All Tasks: {planned_tasks}
    - Future Tasks That Will Be Executed: {future_tasks}
    - Evaluation Metrics: {evaluation_metric}

**Observation**: (this is the result of previous tasks)
Number of executed tasks: {num_task_results}
Executed tasks:

{task_results}

Using the above information, apply instructions to enhance the task and determine specific actions required.
*Note*: You must obey the provided format instruction below:

{format_instructions}
""",
                ),
            ]
        )

    def enhance_task(self, task, state: KaggleProblemState):
        output_parser = PydanticOutputParser(pydantic_object=EnhancedTask)
        format_instructions = output_parser.get_format_instructions()

        response = (self.task_enhancement_prompt | self.llm | output_parser).invoke(
            {
                "current_task": task,
                "problem_description": state.problem_description,
                "dataset_info": str(state.dataset_info),
                "task_results": state.get_task_results(),
                "model_info": str(state.model_info),
                "planned_tasks": str(state.planned_tasks),
                "future_tasks": str(state.get_future_tasks()),
                "evaluation_metric": state.evaluation_metric,
                "best_score": state.best_score,
                "num_task_results": len(state.task_codes_results),
                "format_instructions": format_instructions,
            },
            config=self.config,
        )

        return response

    def __call__(self, state: KaggleProblemState):
        index = state.index + 1

        enhanced_task = self.enhance_task(state.planned_tasks[index], state)

        return {
            "index": index,
            "enhanced_tasks": [enhanced_task],
        }
