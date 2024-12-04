import logging
from typing import Any, Dict

from langchain_core.output_parsers.string import StrOutputParser
from langchain_openai import ChatOpenAI

from prompts.task_enhancer import TASK_ENHANCEMENT_PROMPT
from states.enhancer import EnhancedTask
from states.main import KaggleProblemState
from states.memory import MemoryAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class KaggleTaskEnhancer:
    def __init__(
        self, config: Dict[str, Any], llm: ChatOpenAI, memory_agent: MemoryAgent
    ):
        self.config = config
        self.llm = llm
        self.memory_agent = memory_agent
        self.task_enhancement_prompt = TASK_ENHANCEMENT_PROMPT
        self.output_parser = StrOutputParser()

        logger.info("KaggleTaskEnhancer initialized")

    def enhance_task(self, state: KaggleProblemState) -> Dict[str, Any]:
        current_task = state.planned_tasks[state.index]
        logger.info(f"Enhancing task: {current_task[:100]}...")

        # relevant_context = self.memory_agent.ask_docs(current_task) or []
        # logger.debug(f"Retrieved relevant context: {relevant_context[:5]}...")

        previous_codes = state.get_executed_codes()
        previous_result = state.get_previous_result(
            last_n=3
        )  # Get results of last 2 tasks
        response = self.task_enhancement_prompt.format(
            current_task=current_task,
            evaluation_metrics=state.evaluation_metric,
            problem_description=state.problem_description,
            previous_result=previous_result,
            previous_codes=previous_codes,
            completed_tasks=state.get_executed_tasks(),
            planned_tasks=str(state.planned_tasks),
            future_tasks=str(state.get_future_tasks()),
            # relevant_context=relevant_context,
        )
        print
        logger.debug("Invoking LLM for task enhancement with CoT reasoning")
        self.llm.model_name = "gpt-4o"
        print(self.llm.model_name)
        parsed_task = self.llm.with_structured_output(EnhancedTask).invoke(response)

        logger.info("Task enhanced successfully with CoT reasoning")
        logger.debug(f"Enhanced task: {str(parsed_task)[:100]}...")

        # Add the enhanced task to memory
        self.memory_agent.add_to_short_term_memory(str(parsed_task))
        logger.debug("Enhanced task with CoT reasoning added to short-term memory")
        print("Parsed Task: \n\n", str(parsed_task))
        return {
            "index": state.index,
            "enhanced_tasks": [parsed_task],
        }

    def __call__(self, state: KaggleProblemState):
        logger.info(f"Processing task at index {state.index}")
        state.index += 1
        return self.enhance_task(state)
