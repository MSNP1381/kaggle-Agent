import logging
from typing import Any, Dict

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_openai import ChatOpenAI

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
        self.task_enhancement_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._get_system_message()),
                ("human", self._get_human_message()),
            ]
        )
        self.output_parser = StrOutputParser()

        logger.info("KaggleTaskEnhancer initialized")

    def _get_system_message(self) -> str:
        return """
You are an AI assistant specializing in enhancing tasks for Kaggle machine learning problems. Your goal is to refine tasks and provide insights to improve the effectiveness of ML notebooks.

Key Responsibilities:
1. Analyze tasks within the project context and suggest improvements.
2. Provide actionable insights based on previous results and project context.
3. Suggest alternative approaches when appropriate.
4. Use concise Chain of Thought (CoT) reasoning to explain your decisions.

Important Notes:
- Balance structure with flexibility for creative problem-solving.
- Consider resources, time constraints, and scalability.
- Address ethical considerations, including data privacy and bias mitigation.
- If crucial information is missing, briefly note what's needed.
- Focus on the most impactful suggestions for the current project stage.
"""

    def _get_human_message(self) -> str:
        return """Task: {current_task}

Context:
- Problem: {problem_description}
- Metrics: {evaluation_metrics}
- Project State:

```{completed_tasks}```

Current:

```{current_task}```


Planned:

```{future_tasks}```

Previous Result: ```{previous_result}```
Previous Codes: ```{previous_codes}```
- Relevant Context: {relevant_context}

Instructions:
Analyze the task and provide enhancements using the following structure:

1. Task Analysis (2-3 sentences, use CoT):
   - Key components and relation to project goals
   - Potential challenges and opportunities

2. Insights from Previous Work (2-3 sentences, use CoT):
   - Relevant learnings from past tasks
   - Applicability to current task

3. Top 3 Suggested Enhancements (1-2 sentences each, use CoT):
   - Prioritize based on potential impact
   - Explain reasoning and expected outcomes

4. Risks and Ethical Considerations (1-2 sentences, use CoT):
   - Identify key risks or ethical issues
   - Suggest mitigation strategies

5. Alternative Approach (if applicable, 1-2 sentences, use CoT):
   - Briefly describe an alternative method
   - Compare pros and cons to suggested enhancements

6. Confidence and Uncertainties (1 sentence):
   - Express confidence level in suggestions (High/Medium/Low)
   - Note any significant uncertainties
7. write your final answer in mardown format in wrapped inside three back ticks "```"
Keep the total response under 600 words. Prioritize clarity and actionability in your suggestions. If crucial information is missing, briefly note what additional details would be helpful.
"""

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
            completed_tasks=state.get_planned_tasks(),
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
