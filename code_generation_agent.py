import ast
import operator
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from typing import Annotated, List, TypedDict
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from states.code import Code
from states.main import KaggleProblemState
from prompts.code_generation_prompt import (
    IMPROVED_CODE_GEN_PROMPT,
    VARIABLE_CORRECTION_PROMPT,
)
from states.memory import MemoryAgent
import re
import logging
from utils import CellError, NotebookExecutorInterface, cc, exec2s
from injector import inject

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("code_generation_agent.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def remove_color(text: str) -> str:
    """
    Clean the text by removing ANSI color codes, escape sequences, and other special characters.
    This function is particularly useful for cleaning stdout and error messages from Python exceptions.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    # Remove ANSI escape sequences
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    text = ansi_escape.sub("", text)

    # Remove other escape sequences
    text = re.sub(r"\x1b\[\d+;\d+m", "", text)
    text = re.sub(r"\x1b\[\d+m", "", text)

    # Remove backspace characters
    text = re.sub(r"\b", "", text)

    # Remove carriage returns and line feeds
    text = text.replace("\r", "").replace("\n", " ")

    # Remove extra whitespace
    text = " ".join(text.split())

    return text


class GeneratedCode(BaseModel):
    """Code output"""

    imports: str = Field(description="The import statements for the code")
    code: str = Field(description="The main code block")
    description: str = Field(
        description="description of this code",
        default="",  # Optional field with default empty string
    )

    def __str__(self) -> str:
        return self.imports + "\n" + self.code


class CodeGraphState(TypedDict):
    error: str
    erro_msg: str
    generation: GeneratedCode
    iterations: int
    kaggle_state: KaggleProblemState
    ast_variables: set[str]
    result: str
    messages: Annotated[List, operator.add]


class ReActStep(BaseModel):
    thought: str = Field(description="The agent's reasoning about the current situation")
    action: str = Field(description="The action to be taken based on the thought")
    action_input: dict = Field(description="Input parameters for the action")


class CodeGenerationAgent:
    @inject
    def __init__(
        self,
        config,
        proxy,
        nb_executor: NotebookExecutorInterface,
        memory_agent: MemoryAgent,
        model="gpt-4o-mini",
        max_iterations=3,
        base_url="https://api.avalai.ir/v1",
    ):
        self.base_url = base_url
        self.proxy = proxy
        self.max_iterations = max_iterations
        self.config = config
        self.nb_executor = nb_executor
        self.code_gen_prompt = IMPROVED_CODE_GEN_PROMPT
        self.variable_correction_prompt = VARIABLE_CORRECTION_PROMPT
        self.output_parser = PydanticOutputParser(pydantic_object=GeneratedCode)
        self.format_instructions = self.output_parser.get_format_instructions()
        self.llm = ChatOpenAI(
            base_url=base_url, model=model, http_client=proxy, temperature=0
        ).with_structured_output(schema=GeneratedCode)
        self.uploaded = False
        self.code_gen_chain = self.code_gen_prompt | self.llm
        self.workflow = self.create_workflow()
        self.memory_agent = memory_agent

    def add_initial_data_to_memory(self, state: KaggleProblemState):
        # Add data utils output to long-term memory
        data_utils_content = {
            "dataset_info": state.dataset_info,
            "quantitative_analysis": getattr(state, "quantitative_analysis", None),
            "qualitative_analysis": getattr(state, "qualitative_analysis", None),
        }
        # Only add feature_recommendations if it exists
        if hasattr(state, "feature_recommendations"):
            data_utils_content["feature_recommendations"] = (
                state.feature_recommendations
            )

        self.memory_agent.add_to_long_term_memory(
            {"type": "data_utils", "content": data_utils_content}
        )

        # Add scraper output to long-term memory
        self.memory_agent.add_to_long_term_memory(
            {
                "type": "scraper",
                "content": {
                    "problem_description": state.problem_description,
                    "evaluation_metric": state.evaluation_metric,
                },
            }
        )
        self.memory_agent.add_to_short_term_memory(state.problem_description)
        self.memory_agent.add_to_short_term_memory(state.get_task_results())

    def react(self, state: CodeGraphState) -> ReActStep:
        kaggle_state = state["kaggle_state"]
        current_task = str(kaggle_state.enhanced_tasks[kaggle_state.index])
        relevant_context = self.memory_agent.get_relevant_context(current_task)

        prompt = f"""
        Given the current task: {current_task}
        And the relevant context: {relevant_context}

        Think about the next step to take in solving this Kaggle problem.
        Then decide on an action to take.

        Respond in the following format:
        Thought: [Your reasoning about the current situation]
        Action: [The action to take. Choose from: GenerateCode, ExecuteCode, AnalyzeError, Finish]
        Action Input: [A JSON object with parameters for the action]

        Remember to consider the current state of the problem, any previous code generated, and potential errors or improvements.
        """
        response = self.llm.invoke(prompt)
        return ReActStep.model_validate_json(response)

    def generate_code(self, state: CodeGraphState, action_input: dict) -> CodeGraphState:
        logger.info("---GENERATING CODE SOLUTION---")
        kaggle_state = state["kaggle_state"]
        iterations = state["iterations"]
        messages = state["messages"]

        try:
            code_solution: GeneratedCode = self.code_gen_chain.invoke(action_input, config=self.config)

            self.memory_agent.add_to_short_term_memory(code_solution.code)
            self.memory_agent.add_to_long_term_memory({
                "task": action_input["current_task"],
                "code": code_solution.code,
                "description": code_solution.description,
            })

            return {
                **state,
                "generation": code_solution,
                "iterations": iterations,
                "error": "no",
                "messages": messages + [("ai", str(code_solution))],
            }
        except Exception as e:
            logger.error(f"Error during code generation: {e}", exc_info=True)
            return {
                **state,
                "generation": GeneratedCode(imports="", code="", description=""),
                "iterations": iterations + 1,
                "error": "yes",
                "erro_msg": str(e),
                "messages": messages,
            }

    def execute_code(self, state: CodeGraphState) -> CodeGraphState:
        logger.info("---EXECUTING CODE---")
        code_solution = state["generation"]
        iterations = state["iterations"]

        try:
            result = self.nb_executor.test_and_execute(str(code_solution))
            return {
                **state,
                "result": exec2s(result),
                "error": "no",
            }
        except CellError as e:
            logger.warn(f"❌ ---CODE EXECUTION FAILED: {e.ename}--- ❌")
            return {
                **state,
                "error": "yes",
                "erro_msg": e.traceback,
                "iterations": iterations + 1,
            }

    def analyze_error(self, state: CodeGraphState, action_input: dict) -> CodeGraphState:
        logger.info("---ANALYZING ERROR---")
        error_msg = state["erro_msg"]
        
        prompt = f"""
        Analyze the following error and suggest a fix:
        {error_msg}

        Consider the current task: {action_input['current_task']}
        And the current code:
        {state['generation']}

        Provide your analysis and suggestion in the following format:
        Analysis: [Your understanding of the error]
        Suggestion: [Your suggested fix]
        """

        response = self.llm.predict(prompt)
        return {
            **state,
            "error_analysis": response,
        }

    def create_workflow(self):
        workflow = StateGraph(CodeGraphState)

        workflow.add_node("react", self.react)
        workflow.add_node("generate_code", self.generate_code)
        workflow.add_node("execute_code", self.execute_code)
        workflow.add_node("analyze_error", self.analyze_error)

        workflow.add_edge("react", "generate_code")
        workflow.add_edge("generate_code", "execute_code")
        workflow.add_edge("execute_code", "react")
        workflow.add_edge("analyze_error", "react")

        workflow.set_entry_point("react")

        workflow.add_conditional_edges(
            "react",
            lambda x: x.action,
            {
                "GenerateCode": "generate_code",
                "ExecuteCode": "execute_code",
                "AnalyzeError": "analyze_error",
                "Finish": END,
            }
        )

        return workflow.compile()

    def __call__(self, state: KaggleProblemState):
        initial_state = {
            "kaggle_state": state,
            "iterations": 0,
            "generation": GeneratedCode(imports="", code="", variables={}),
            "messages": [],
        }

        if not self.uploaded:
            self._cp_files(state)
            self.uploaded = True
            self.add_initial_data_to_memory(state)

        result = self.workflow.invoke(initial_state, config=self.config)

        task_codes = state.task_codes_results
        task_codes.append(
            (
                state.enhanced_tasks[state.index],
                Code(
                    imports=result["generation"].imports,
                    code=result["generation"].code,
                    description=result["generation"].description,
                ),
                str(result.get("result", "")),
            )
        )

        return {
            "task_codes_results": task_codes,
        }

    def _cp_files(self, state: KaggleProblemState):
        # self.dataset_path = "./ongoing/train.csv"
        # self.test_dataset_path = "./ongoing/test.csv"

        with open(state.dataset_path, "rb") as f:
            self.memory_agent.add_to_short_term_memory(
                f"{{dataset_path :  { state.dataset_path} }}"
            )
            env_var = self.nb_executor.upload_file_env(f)
        with open(state.test_dataset_path, "rb") as f:
            self.memory_agent.add_to_short_term_memory(
                f"{{test_dataset_path :  { state.test_dataset_path} }}"
            )
            env_var = self.nb_executor.upload_file_env(f)