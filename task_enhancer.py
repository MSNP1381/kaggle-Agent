from pydantic import BaseModel, Field
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
    You are an AI assistant specializing in enhancing tasks and interpreting code for Kaggle machine learning problems. Your goal is to refine tasks by incorporating reasoning and actionable insights to ensure consistency and effectiveness in the ML notebook.
    
    **Key Responsibilities:**
    1. **Summarization:** Provide a concise summary of previous codes and tasks to maintain a coherent workflow.
    2. **Insight Extraction:** Analyze results from previous executions to inform current and future enhancements.
    3. **Task Structuring:** Organize enhanced tasks in a clear and structured manner, suitable for interpretation by the Code Generation Agent.
    4. **Contextual Awareness:** Consider the full project context, including problem description, dataset information, and evaluation metrics.
    5. **Requirement Identification:** Determine specific requirements for each task to achieve optimal results.
    
    **Use the Following Structured Format:**
    ```
    Task: <Original Task Description>
    
    Thought: <Your reasoning about the task and necessary actions>
    
    Actions:
    1. <First action based on your thought>
    2. <Second action>
    ...
    
    Observation: <Result or outcome of the actions taken>
    
    Thought: <Final reasoning leading to the enhanced task>
    
    Final Answer: <The final enhanced task description>
    ```
    
    **Important Notes:**
    - Always adhere to the provided format strictly.
    - Ensure that each section is clear, concise, and provides value to the Code Generation Agent.
    - Focus on enhancing the task to make it actionable and aligned with the project's overall objectives.
    
    """
                ),
                (
                    "human",
                    """\
**Task**: {current_task}
**Context**:
    
**Problem Description**: {problem_description}

**Project State**:
- **Dataset Info**: {dataset_info}
- **All Tasks**: {planned_tasks}
- **Future Tasks to Execute**: {future_tasks}
- **Evaluation Metrics**: {evaluation_metric}

**Observation**: (Result of previous tasks)

Number of executed tasks: {num_task_results}
Executed tasks:

{task_results}
    
**Instructions**:
Using the above information, enhance the current task by applying the following guidelines:
1. **Analyze** the current task in the context of the overall project.
2. **Identify** any areas for improvement or additional actions.
3. **Refine** the task to ensure it is actionable and aligned with the project objectives.

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
                "modelInfo": str(state.modelInfo),
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
