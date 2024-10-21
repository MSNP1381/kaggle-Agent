import logging
from typing import Any, Dict

from langchain_core.output_parsers.string import StrOutputParser
from langchain_openai import ChatOpenAI

from prompts.task_enhancer import TASK_ENHANCEMENT_PROMPT
from states.enhancer import EnhancedTask
from states.main import KaggleProblemState
from states.memory import MemoryAgent
from utils import extract_markdown

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

        relevant_context = self.memory_agent.ask_docs(current_task) or []
        logger.debug(f"Retrieved relevant context: {relevant_context[:5]}...")

        previous_codes = state.get_executed_codes()
        previous_result = state.get_previous_result(
            last_n=3
        )  # Get results of last 2 tasks
        response = self.task_enhancement_prompt.format_messages(
            current_task=current_task,
            evaluation_metrics=state.evaluation_metric,
            problem_description=state.problem_description,
            previous_result=previous_result,
            previous_codes=previous_codes,
            completed_tasks=state.get_executed_tasks(),
            planned_tasks=str(state.planned_tasks),
            future_tasks=str(state.get_future_tasks()),
            relevant_context=relevant_context,
        )

        logger.debug("Invoking LLM for task enhancement with CoT reasoning")
        enhanced_task = (self.llm | self.output_parser).invoke(response)
        parsed_task = EnhancedTask(final_answer=enhanced_task)
        parsed_task.final_answer = extract_markdown(parsed_task.final_answer)
        logger.info("Task enhanced successfully with CoT reasoning")
        logger.debug(f"Enhanced task: {parsed_task.final_answer[:100]}...")

        # Add the enhanced task to memory
        self.memory_agent.add_to_short_term_memory(str(parsed_task))
        logger.debug("Enhanced task with CoT reasoning added to short-term memory")

        return {
            "index": state.index,
            "enhanced_tasks": state.enhanced_tasks + [parsed_task],
        }

    def __call__(self, state: KaggleProblemState):
        logger.info(f"Processing task at index {state.index}")
        state.index += 1
        return self.enhance_task(state)
