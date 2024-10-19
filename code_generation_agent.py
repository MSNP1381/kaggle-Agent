import json
import os

from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph

from typing import List, TypedDict

from langchain_core.output_parsers.string import StrOutputParser

from pydantic import BaseModel, Field

from states.code import Code

from states.enhancer import EnhancedTask
from states.main import KaggleProblemState

from prompts.code_generation_prompt import (
    IMPROVED_CODE_GEN_PROMPT,
    DEBUGGING_PROMPT,
    NEW_SOLUTION_PROMPT,
    pkg_str,
)

from states.memory import MemoryAgent
import re
import logging

from utils import (
    CellError,
    NotebookExecutorInterface,
    NotebookFailError,
    exec2s,
    extract_code,
    extract_text_up_to_code,
)

from injector import inject

# Initialize logging

logger = logging.getLogger(__name__)


class DebugCode(BaseModel):
    debug_analysis: str = Field(description="Debug analysis for code generation")

    suggested_fix: str = Field(description="Suggested fix for code generation")


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

    code: str = Field(description="The imports and code")

    def __str__(self) -> str:
        return self.code


class CodeGraphState(TypedDict):
    error: str

    error_name: str

    error_msg: str

    generation: GeneratedCode

    iterations: int

    kaggle_state: KaggleProblemState

    result: str

    messages: List

    analysis: str

    suggested_fix: str


class CodeGenerationAgent:
    @inject
    def __init__(
        self,
        llm: ChatOpenAI,
        config,
        # proxy,
        nb_executor: NotebookExecutorInterface,
        memory_agent: MemoryAgent,
        model="gpt-4o-mini",
        max_iterations=1,
    ):
        self.base_url = os.getenv("BASE_URL", "https://api.avalapis.ir/v1")

        # self.proxy = proxy

        self.max_iterations = max_iterations

        self.config = config

        self.nb_executor = nb_executor

        self.code_gen_prompt = IMPROVED_CODE_GEN_PROMPT

        # self.output_parser = PydanticOutputParser(pydantic_object=GeneratedCode)

        self.output_parser = StrOutputParser()
        # self.format_instructions = self.output_parser.get_format_instructions()

        self.llm_raw = llm

        self.uploaded = False

        self.code_gen_chain = self.code_gen_prompt | self.llm_raw | self.output_parser

        self.workflow = self.create_workflow()

        self.memory_agent = memory_agent

        self.debugging_prompt = DEBUGGING_PROMPT
        self.new_solution_prompt = NEW_SOLUTION_PROMPT

    def add_init_code(self, state: CodeGraphState):
        if len(state["kaggle_state"].task_codes_results) == 0:
            init_code = [
                "import pandas as pd",
                "import numpy as np",
                "train_df = pd.read_csv('./input/train.csv')",
                "test_df = pd.read_csv('./input/test.csv')",
            ]
            prep_code = "\n".join(init_code)
            self.nb_executor.test_and_execute(prep_code)
            enhanced_task = EnhancedTask(
                final_answer="load data from input dricrtory",
            )
            init_code_obj = Code(imports="", code=prep_code, description="")

            state["kaggle_state"].task_codes_results.append(
                (enhanced_task, init_code_obj, "")
            )
            state["kaggle_state"].enhanced_tasks.append(enhanced_task)
            state["kaggle_state"].index += 1

    def choose_base_prompt(self, state: CodeGraphState):
        if state.get("error") == "yes":
            if state["iterations"] <= 1:
                return self.debugging_prompt
            else:
                return self.new_solution_prompt
        else:
            return self.code_gen_prompt

    def generate_code(self, state: CodeGraphState) -> CodeGraphState:
        logger.info("---GENERATING OR DEBUGGING CODE SOLUTION---")

        kaggle_state = state["kaggle_state"]
        iterations = state["iterations"]
        messages = state["messages"]
        current_task = str(kaggle_state.enhanced_tasks[kaggle_state.index - 1])

        # Get context and examples
        relevant_context = self.memory_agent.ask_docs(current_task)
        cot_examples = json.dumps(
            self.memory_agent.get_few_shots(current_task), indent=2
        )
        logger.info(f"COT examples: {str(cot_examples)}")

        evaluation_metric = kaggle_state.evaluation_metric
        planned_tasks = kaggle_state.get_planned_tasks()

        # Choose the appropriate base prompt
        base_prompt = self.choose_base_prompt(state)

        # Add error information to messages if there was an error
        if state.get("error") == "yes":
            error_type = self.categorize_error(state["error_msg"])
            error_message = f"Error Type: {error_type}\nError Message: {state['error_msg']}\nPlease focus on fixing this error in your next code generation."
            messages.append(("human", error_message))

        # Generate code
        code_solution: str = (base_prompt | self.llm_raw | self.output_parser).invoke(
            {
                "pkg_str": pkg_str,
                "problem_description": kaggle_state.problem_description,
                "current_task": current_task,
                "evaluation_metric": evaluation_metric,
                "planned_tasks": planned_tasks,
                "relevant_context": relevant_context,
                "messages": messages,
                "previous_task_results": kaggle_state.get_executed_codes(2),
                "cot_examples": cot_examples,
                "current_code": str(state.get("generation", "")),
                "error_msg": state.get("error_msg", ""),
            },
            config=self.config,
        )

        code_exp = extract_code(code_solution)
        explanation = extract_text_up_to_code(code_solution)
        code_solution = GeneratedCode(code=code_exp)

        logger.info("Generated code solution:")
        logger.info(explanation)

        return {
            "generation": code_solution,
            "iterations": iterations + 1,
            "messages": messages + [("assistant", explanation)],
            "explanation": explanation,
        }

    def execute_code(self, state: CodeGraphState) -> CodeGraphState:
        logger.info("---EXECUTING CODE---")

        code_solution = state["generation"]

        iterations = state["iterations"]

        try:
            result = self.nb_executor.test_and_execute(str(code_solution))
            logger.info(" OOOO ---CODE EXECUTed Successfully :---  OOOO")

            return {
                **state,
                "result": remove_color(exec2s(result)),
                "error": "no",
                "iterations": iterations + 1,
                "messages": [],
            }

        except CellError as e:
            logger.error(f"XXX ---CODE EXECUTION FAILED: {e.ename}--- XXXX")

            return {
                **state,
                "error": "yes",
                "error_msg": remove_color(exec2s(e.evalue)),
                "error_name": remove_color(exec2s(e.ename)),
                "iterations": iterations + 1,
                "messages": [],
            }

    def debug_code(self, state: CodeGraphState) -> CodeGraphState:
        logger.info("---DEBUGGING OR GENERATING NEW SOLUTION---")

        error_msg = state["error_msg"]
        current_code = str(state["generation"])
        current_task = str(
            state["kaggle_state"].enhanced_tasks[state["kaggle_state"].index]
        )
        previous_code = state["kaggle_state"].get_executed_codes(2)

        # Categorize the error
        error_type = self.categorize_error(error_msg)

        # Decide whether to debug or generate a new solution
        if state["iterations"] <= 1 and error_type != "critical":
            approach = "debug"
        else:
            approach = "new_solution"

        prompt = self.create_prompt(
            approach, error_type, current_task, error_msg, current_code, previous_code
        )

        response = self.llm_raw.invoke(prompt)

        if approach == "debug":
            code = extract_code(self.output_parser.invoke(response))
            explanation = extract_text_up_to_code(response)
            return {
                **state,
                "fixed_code": code,
                "error": "yes",
                "messages": [],
                "debug_explanation": explanation,
            }
        else:
            new_solution = extract_code(self.output_parser.invoke(response))
            return {
                **state,
                "generation": GeneratedCode(code=new_solution),
                "error": "no",
                "messages": [],
            }

    def categorize_error(self, error_name: str) -> str:
        # Implement error categorization logic
        # This could use regex patterns or ML techniques to classify errors
        # For now, we'll use a simple placeholder implementation
        if "syntax" in error_name.lower():
            return "syntax"
        elif "runtime" in error_name.lower():
            return "runtime"
        elif "value" in error_name.lower():
            return "value"
        else:
            return "unknown"

    def create_prompt(
        self,
        approach: str,
        error_type: str,
        current_task: str,
        error_msg: str,
        current_code: str,
        previous_code: str,
    ) -> str:
        # Implement prompt creation logic based on the approach and error type
        pass

    def create_workflow(self):
        workflow = StateGraph(CodeGraphState)

        workflow.add_node("generate_code", self.generate_code)

        workflow.add_node("execute_code", self.execute_code)

        # workflow.add_node("debug_code", self.debug_code)

        # workflow.add_node("analyze_error", self.analyze_error)

        workflow.add_edge("generate_code", "execute_code")

        # workflow.add_edge("debug_code", "execute_code")

        workflow.set_entry_point("generate_code")

        def decide(x: CodeGraphState):
            if x["error"] == "yes" and x["iterations"] > self.max_iterations:
                raise NotebookFailError(x["error_msg"], str(x["generation"]))
            if x["error"] == "yes" and x["iterations"] <= self.max_iterations:
                return "debug_code"
            else:
                return END

        workflow.add_conditional_edges(
            "execute_code",
            decide,
            {
                "debug_code": "generate_code",
                END: END,
            },
        )

        return workflow.compile()

    def __call__(self, state: KaggleProblemState, temp=0, max_iterations=1):
        initial_state = {
            "kaggle_state": state,
            "iterations": 0,
            "generation": GeneratedCode(code=""),
            "messages": [],
            "error": "no",
            "error_msg": "",
        }

        self.max_iterations = max_iterations
        self.llm_raw.temperature = temp
        self.code_gen_chain.steps[1].temperature = temp
        self.add_init_code(initial_state)
        result = self.workflow.invoke(initial_state, config=self.config)

        task_codes = state.task_codes_results
        task_code_new = (
            state.enhanced_tasks[state.index],
            Code(imports="", code=result["generation"].code, description=""),
            str(result.get("result", "")),
        )
        task_codes.append(task_code_new)
        self.memory_agent.add_example(*map(str, task_code_new))

        return {
            "task_codes_results": task_codes,
        }
