import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
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
        self.output_parser = StrOutputParser()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info("KaggleTaskEnhancer initialized")

    def _get_system_message(self) -> str:
        return """
You are an AI assistant specializing in enhancing tasks for Kaggle machine learning problems. Your goal is to refine tasks and provide insights to improve the effectiveness of ML notebooks.

Key Responsibilities:
1. Analyze and enhance tasks within the context of the overall project.
2. Provide actionable insights based on previous results and project context.
3. Identify potential risks, alternative approaches, and ethical considerations.
4. Suggest improvements to the overall project plan when appropriate.
5. Explain your reasoning clearly for each suggestion or enhancement.

Important Notes:
- Balance structure with flexibility to allow for creative problem-solving.
- Consider computational resources, time constraints, and scalability in your suggestions.
- Address ethical considerations, including data privacy and bias mitigation.
- If crucial information is missing, request it before proceeding.
"""

    def _get_human_message(self) -> str:
        return """Task: {current_task}

Context:
- Problem Description: {problem_description}
- Evaluation Metrics: {evaluation_metrics}

Project State:
- Completed Tasks: {completed_tasks}
- Current Task: {current_task}
- Planned Future Tasks: {future_tasks}

Previous Results:
{previous_results}

Relevant Context:
{relevant_context}

Instructions:
1. Analyze the current task in the context of the overall project.
2. Suggest enhancements or modifications to the task, explaining your reasoning.
3. Identify any potential risks or ethical considerations.
4. Propose alternative approaches if applicable.
5. Suggest any necessary changes to the overall project plan.
6. Highlight any assumptions you're making in your analysis.
7. If crucial information is missing, specify what additional details you need.

Please structure your response as follows:
1. Task Analysis
2. Suggested Enhancements
3. Potential Risks and Ethical Considerations
4. Alternative Approaches
5. Project Plan Modifications
6. Assumptions and Information Requests

Your goal is to provide clear, actionable insights that will improve the ML project's effectiveness and address potential challenges. Be specific in your suggestions and explain how they relate to the project's objectives and constraints.
"""

    def enhance_task(self, state: KaggleProblemState) -> Dict[str, Any]:
        current_task = state.planned_tasks[state.index]
        self.logger.info(f"Enhancing task: {current_task}")

        relevant_context = self.memory_agent.ask_docs(current_task)
        self.logger.debug(f"Retrieved relevant context: {relevant_context[:5]}...")

        response = self.task_enhancement_prompt.format_messages(
            current_task=current_task,
            evaluation_metrics=state.evaluation_metric,
            problem_description=state.problem_description,
            previous_results=self.memory_agent.results_summary,
            task_results=state.get_executed_codes(2),
            completed_tasks=state.get_planned_tasks(),
            planned_tasks=str(state.planned_tasks),
            future_tasks=str(state.get_future_tasks()),
            num_task_results=len(state.task_codes_results),
            relevant_context=relevant_context,
        )

        self.logger.debug("Invoking LLM for task enhancement")
        enhanced_task = (self.llm | self.output_parser).invoke(response)
        parsed_task = EnhancedTask(final_answer=enhanced_task)

        self.logger.info("Task enhanced successfully")
        self.logger.debug(f"Enhanced task: {parsed_task.final_answer[:100]}...")

        # Add the enhanced task to memory
        self.memory_agent.add_to_short_term_memory(str(parsed_task))
        self.logger.debug("Enhanced task added to short-term memory")

        return {
            "index": state.index,
            "enhanced_tasks": state.enhanced_tasks + [parsed_task],
        }

    def __call__(self, state: KaggleProblemState):
        self.logger.info(f"Processing task at index {state.index}")
        state.index += 1
        return self.enhance_task(state)
