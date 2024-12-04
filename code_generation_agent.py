import logging
import re
from typing import List, TypedDict
from langchain.prompts import ChatPromptTemplate
from injector import inject
from langchain_core.output_parsers.string import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from prompts.code_generation_prompt import (
    DEBUGGING_PROMPT,
    IMPROVED_CODE_GEN_PROMPT,
    NEW_SOLUTION_PROMPT,
    pkg_str,
)
from states.code import Code
from states.main import KaggleProblemState
from states.memory import MemoryAgent
from utils import (
    CellError,
    NotebookExecutorInterface,
    NotebookFailError,
    exec2s,
    extract_code,
    extract_text_up_to_code,
)

# Initialize logging

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
    """python code for current task"""

    code: str = Field(description="The python code")

    def __str__(self) -> str:
        return self.code


class CodeGraphState(TypedDict):
    error: str

    error_name: str

    error_msg: str
    error_code: str
    generation: GeneratedCode

    iterations: int

    kaggle_state: KaggleProblemState

    result: str

    messages: List

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
        max_iterations=1,
    ):
        # self.proxy = proxy

        self.max_iterations = max_iterations

        self.config = config

        self.nb_executor = nb_executor

        self.code_gen_prompt = IMPROVED_CODE_GEN_PROMPT

        self.output_parser = StrOutputParser()
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.llm_raw = llm

        self.code_gen_chain = self.code_gen_prompt | self.llm_raw | self.output_parser

        self.workflow = self.create_workflow()

        self.memory_agent = memory_agent

        self.debugging_prompt = DEBUGGING_PROMPT
        self.new_solution_prompt = NEW_SOLUTION_PROMPT

        # Define the OpenAI function for Kaggle submission

        # Add the Kaggle submission function to the LLM's available functions

    def choose_base_prompt(self, state: CodeGraphState):
        if state.get("error") == "yes":
            if state["iterations"] <= 3:
                logger.info("---DEBUGGING PROMPT Selected---")
                return self.debugging_prompt

            else:
                logger.info("---NEW SOLUTION PROMPT Selected---")
                return self.new_solution_prompt
        else:
            logger.info("---CODE GENERATION PROMPT Selected---")
            return self.code_gen_prompt.partial(
                history=state["kaggle_state"].get_history(2)
            )

    def generate_code(self, state: CodeGraphState) -> CodeGraphState:
        logger.info("---GENERATING OR DEBUGGING CODE SOLUTION---")

        kaggle_state = state["kaggle_state"]
        iterations = state["iterations"]

        current_task = str(kaggle_state.enhanced_tasks[-1])
        plan = kaggle_state.planned_tasks
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        task_formatted = f"""For the following plan:
                {plan_str}\n\nYou are tasked with executing step {1}, {current_task}."""
        # relevant_context = self.memory_agent.ask_docs(current_task)
        # few_shots_examples = json.dumps(
        #     self.memory_agent.get_few_shots(current_task), indent=2
        # )
        relevant_context = ""
        few_shots_examples = ""
        logger.info(f"Few shots examples: {str(few_shots_examples)[:100]}...")

        evaluation_metric = kaggle_state.evaluation_metric
        planned_tasks = kaggle_state.get_executed_tasks()

        base_prompt = self.choose_base_prompt(state)

        code_solution: GeneratedCode = (
            base_prompt | self.llm_raw.with_structured_output(GeneratedCode)
        ).invoke(
            {
                "pkg_str": pkg_str,
                "problem_description": kaggle_state.problem_description,
                "current_task": task_formatted,
                "evaluation_metric": evaluation_metric,
                "planned_tasks": planned_tasks,
                "error_code": state.get("error_code", ""),
                "previous_tasks": kaggle_state.get_previous_result(2),
                "relevant_context": relevant_context,
                "history": kaggle_state.get_history(2),
                "previous_codes": kaggle_state.get_executed_codes(2),
                "few_shots_examples": few_shots_examples,
                "current_code": str(state.get("generation", "")),
                "error_msg": state.get("error_msg", ""),
                "suggested_fix": state.get("suggested_fix", ""),
                "reasoning": state.get("suggested_fix", ""),
            },
            config=self.config,
        )

        # code_solution = GeneratedCode(code=code_exp)

        logger.info("Generated code solution:")
        logger.info(code_solution.code)
        return {
            "generation": code_solution,
            "iterations": iterations + 1,
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
            logger.error(f"XXX ---CODE EXECUTION FAILED: {e.evalue}--- XXXX")

            return {
                **state,
                "error": "yes",
                "error_msg": remove_color(exec2s(e.traceback)),
                "error_name": remove_color(exec2s(e.ename)),
                "error_code": str(code_solution),
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

    def reflect_on_error(self, state: CodeGraphState):
        code = state["generation"]
        error = state["error_msg"]
        system = (
            "system",
            "you are an expert in finding errors in code and debugging the code\n\nyour job is to do reasoning about error and explain why error has happend in a way that is underestandable to machine",
        )
        user = ("user", "code is : \n\n {code} \n------\n error is : \n\n {error}")

        error_resoning = (
            ChatPromptTemplate.from_messages([system, user]) | self.llm_raw
        ).invoke({"code": code, "error": error}, config=self.config)
        return {"suggested_fix": error_resoning}

    def create_workflow(self):
        workflow = StateGraph(CodeGraphState)

        workflow.add_node("generate_code", self.generate_code)

        workflow.add_node("execute_code", self.execute_code)

        workflow.add_node("reflect_on_error", self.reflect_on_error)

        # workflow.add_node("analyze_error", self.analyze_error)
        workflow.set_entry_point("generate_code")

        workflow.add_edge("generate_code", "execute_code")

        workflow.add_edge("reflect_on_error", "generate_code")

        def decide(x: CodeGraphState):
            if x["error"] == "yes" and x["iterations"] > self.max_iterations:
                raise NotebookFailError(x["error_msg"], str(x["generation"]))
            if x["error"] == "yes" and x["iterations"] <= self.max_iterations:
                return "reflect_on_error"
            else:
                return END

        workflow.add_conditional_edges(
            "execute_code",
            decide,
            [
                "reflect_on_error",
                END,
            ],
        )

        return workflow.compile()

    def __call__(self, state: KaggleProblemState, max_iterations=4):
        initial_state = {
            "kaggle_state": state,
            "iterations": 0,
            "generation": GeneratedCode(code=""),
            "messages": [],
            "error": "no",
            "error_msg": "",
        }

        self.max_iterations = max_iterations

        # self.add_init_code(initial_state)
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
