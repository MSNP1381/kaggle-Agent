import httpx
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from code_generation_agent import CodeGenerationAgent, Code
from typing import Dict, Any, Union, List

class KaggleProblemPlanner:
    def __init__(self, config, proxy, nb_executor):
        self.config = config
        self.nb_executor = nb_executor
        self.llm = ChatOpenAI(model="gpt-4o-mini", http_client=proxy, temperature=0)
        self.code_generation_agent = CodeGenerationAgent(config=config, proxy=proxy, nb_executor=self.nb_executor)
        
        self.planner_prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant tasked with solving a Kaggle machine learning problem.
        Given the problem description and current state, create or update a plan to solve the problem.
        You must follow Notes below for code generation

        Problem Description:
        {problem_description}

        Current State:
        {state}

        Your task is to:
        1. Analyze the problem description and current state.
        2. Create or update a plan of tasks to solve the Kaggle problem.
        3. Ensure the plan covers all necessary steps: preprocessing, feature engineering, model selection, training, and evaluation.
        
        Notes:
        - Ensure don't call plt.show() method for plots in codes
        - Don't use any visually formatted like rendered HTML outputs, just use text or dict for outputs
        - Consider the current state, including previous task results and existing variables

        Respond with a list of planned tasks, each on a new line.
        """)

        self.code_generation_initial_prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant tasked with generating Python code for a Kaggle machine learning problem.
        We will have a conversation about the code generation process, where you will generate code incrementally and I will provide feedback or ask for modifications.

        Problem Description:
        {problem_description}

        Current Task: {current_task}

        Project State:
        {project_state}

        Previous Code:
        {previous_code}

        Previous Results:
        {previous_results}

        Your task is to:
        1. Analyze the problem description, current state, and task.
        2. Propose an initial approach to accomplish the current task.
        3. Begin generating Python code that implements this approach.
        4. Explain your thinking and ask for clarification if needed.

        Notes:
        - Ensure don't call plt.show() method for plots in codes
        - Don't use any visually formatted like rendered HTML outputs, just use text or dict for outputs
        - Make use of libraries like pandas, numpy, scikit-learn, and others as appropriate
        - Comment your code to explain key steps and how you're using previous results or code
        - If using previous results, clearly indicate how they're being incorporated

        Start by proposing your initial approach and the first few lines of code.
        """)

    def plan(self, state):
        response = self.llm.invoke(self.planner_prompt.format_messages(
            problem_description=state.problem_description,
            state=str(state.__dict__)
        ), config=self.config)
        return response.content.strip().split("\n")

    def generate_code(self, state: 'KaggleProblemState') -> Code:
        previous_code = self._get_relevant_previous_code(state)
        previous_results = self._get_relevant_previous_results(state)
        
        conversation = [
            {"role": "system", "content": self.code_generation_initial_prompt.format(
                problem_description=state.problem_description,
                current_task=self._get_task_description(state.current_task),
                project_state=self._format_project_state(state),
                previous_code=previous_code,
                previous_results=previous_results
            )}
        ]

        full_code = ""
        
        while True:
            response = self.llm.invoke(conversation, config=self.config)
            conversation.append({"role": "assistant", "content": response.content})
            
            # Extract code from the response
            code_lines = self._extract_code(response.content)
            full_code += "\n".join(code_lines) + "\n"
            
            # Check if the code is complete
            if self._is_code_complete(full_code, state.current_task):
                break
            
            # If not complete, ask for the next part
            conversation.append({
                "role": "human",
                "content": "The code looks good so far. Please continue and provide the next part of the implementation."
            })

        return Code(full_code)

    def _get_relevant_previous_code(self, state: 'KaggleProblemState') -> str:
        relevant_code = []
        for task, code in state.task_codes.items():
            if self._is_relevant(state.current_task, task):
                relevant_code.append(f"# Code for task: {self._get_task_description(task)}\n{code}\n")
        return "\n".join(relevant_code)

    def _get_relevant_previous_results(self, state: 'KaggleProblemState') -> str:
        relevant_results = []
        for task, result in state.task_results.items():
            if self._is_relevant(state.current_task, task):
                relevant_results.append(f"# Result for task: {self._get_task_description(task)}\n{result}\n")
        return "\n".join(relevant_results)

    def _is_relevant(self, current_task: Union[str, 'EnhancedTask'], previous_task: Union[str, 'EnhancedTask']) -> bool:
        current_task_desc = self._get_task_description(current_task)
        previous_task_desc = self._get_task_description(previous_task)
        
        current_keywords = set(current_task_desc.lower().split())
        previous_keywords = set(previous_task_desc.lower().split())
        return len(current_keywords.intersection(previous_keywords)) > 0

    @staticmethod
    def _get_task_description(task: Union[str, 'EnhancedTask']) -> str:
        if isinstance(task, str):
            return task
        elif hasattr(task, 'task'):
            return task.task
        else:
            return str(task)

    def _format_project_state(self, state: 'KaggleProblemState') -> str:
        formatted_state = {
            "dataset_info": state.dataset_info,
            "previous_tasks": [self._get_task_description(task) for task in state.previous_tasks],
            "model_info": state.model_info,
            "evaluation_metric": state.evaluation_metric,
            "best_score": state.best_score
        }
        return str(formatted_state)

    def _extract_code(self, response: str) -> List[str]:
        # Simple extraction of code blocks
        code_blocks = []
        in_code_block = False
        for line in response.split('\n'):
            if line.strip().startswith('```python'):
                in_code_block = True
                continue
            elif line.strip() == '```' and in_code_block:
                in_code_block = False
                continue
            if in_code_block:
                code_blocks.append(line)
        return code_blocks

    def _is_code_complete(self, code: str, task: Union[str, 'EnhancedTask']) -> bool:
        # This is a simplified check. You might want to implement a more sophisticated one.
        task_desc = self._get_task_description(task).lower()
        if 'model' in task_desc and 'fit' in code:
            return True
        if 'preprocess' in task_desc and 'transform' in code:
            return True
        if 'evaluate' in task_desc and 'score' in code:
            return True
        # Default to False if we're not sure
        return False