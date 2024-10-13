from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

from states.enhancer import EnhancedTask
from states.main import KaggleProblemState
from states.memory import MemoryAgent


class KaggleTaskEnhancer:
    def __init__(
        self, config: Dict[str, Any], llm: ChatOpenAI, memory_agent: MemoryAgent
    ):
        self.config = config
        self.llm = llm
        self.memory_agent = memory_agent
        self.task_enhancement_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._get_system_message()),
                ("human", self._get_human_message()),
            ]
        )
        self.output_parser = PydanticOutputParser(pydantic_object=EnhancedTask)

    def _get_system_message(self) -> str:
        return """You are an AI assistant specializing in enhancing tasks and interpreting code for Kaggle machine learning problems. Your goal is to refine tasks by incorporating reasoning and actionable insights to ensure consistency and effectiveness in the ML notebook.

**Key Responsibilities:**
1. **Summarization:** Provide a concise summary of previous codes and tasks to maintain a coherent workflow.
2. **Insight Extraction:** Analyze results from previous executions to inform current and future enhancements.
3. **Task Structuring:** Organize enhanced tasks in a clear and structured manner, suitable for interpretation by the Code Generation Agent.
4. **Contextual Awareness:** Consider the full project context, including problem description, dataset information, and evaluation metrics.
5. **Requirement Identification:** Determine specific requirements for each task to achieve optimal results.


**Important Notes:**
- Always adhere to the provided format strictly.
- Ensure that each section is clear, concise, and provides value to the Code Generation Agent.
- Focus on enhancing the task to make it actionable and aligned with the project's overall objectives.
"""

    def _get_human_message(self) -> str:
        return """**Task**: {current_task}
**Context**:

**Problem Description**: {problem_description}

**Project State**:
- **All Tasks**: {planned_tasks}
- **Future Tasks to Execute**: {future_tasks}

**Observation**: (Result of previous tasks)

Number of executed tasks: {num_task_results}
Executed tasks:

{task_results}

**Relevant Context**:
{relevant_context}

**Instructions**:
Using the above information, enhance the current task by applying the following guidelines:
1. **Analyze** the current task in the context of the overall project.
2. **Identify** any areas for improvement or additional actions.
3. **Refine** the task to ensure it is actionable and aligned with the project objectives.

Using the above information, apply instructions to enhance the task and determine specific actions required.
*Note*: You must obey the provided format instruction below:

{format_instructions}
"""

    def enhance_task(self, state: KaggleProblemState) -> Dict[str, Any]:
        current_task = state.planned_tasks[state.index]
        relevant_context = self.memory_agent.get_relevant_context(current_task)

        format_instructions = self.output_parser.get_format_instructions()

        response = self.task_enhancement_prompt.format_messages(
            current_task=current_task,
            problem_description=state.problem_description,
            task_results=state.get_task_results(),
            planned_tasks=str(state.planned_tasks),
            future_tasks=str(state.get_future_tasks()),
            num_task_results=len(state.task_codes_results),
            format_instructions=format_instructions,
            relevant_context="\n".join(relevant_context),
        )

        enhanced_task = self.llm.invoke(response)
        parsed_task = self.output_parser.parse(enhanced_task.content)

        # Add the enhanced task to memory
        self.memory_agent.add_to_short_term_memory(str(parsed_task))
        self.memory_agent.add_to_long_term_memory(
            {"type": "enhanced_task", "content": parsed_task.dict()}
        )

        return {
            "index": state.index + 1,
            "enhanced_tasks": [parsed_task],
        }

    def __call__(self, state: KaggleProblemState):
        index = state.index + 1

        # Get relevant context from memory
        relevant_context = self.memory_agent.get_relevant_context(
            state.planned_tasks[index]
        )

        # enhanced_task = self.llm.invoke(
        #     self.task_enhancement_prompt.format(
        #         current_task=state.planned_tasks[index],
        #         problem_description=state.problem_description,
        #         dataset_info=str(state.dataset_info),
        #         task_results=state.get_task_results(),
        #         modelInfo=str(state.modelInfo),
        #         planned_tasks=str(state.planned_tasks),
        #         future_tasks=str(state.get_future_tasks()),
        #         evaluation_metric=state.evaluation_metric,
        #         best_score=state.best_score,
        #         num_task_results=len(state.task_codes_results),
        #         format_instructions=self.memory_agent.get_format_instructions(),
        #         relevant_context="\n".join(relevant_context),
        #     )
        # )
        return self.enhance_task(state)
