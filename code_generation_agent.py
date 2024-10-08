import ast
import json
import operator
import pprint
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

import re
import logging

from utils import CellError, NotebookExecutorInterface, cc, exec2s

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("code_generation_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



def remove_color(clean_text : str) -> str:
    """
    Clean the stdout text by removing ANSI color codes and other unwanted characters.
    """
    clean_text = re.sub(r"\\x1b\[\d+;\d+m~", "", clean_text)
    clean_text = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", clean_text)
    return clean_text


class GeneratedCode(BaseModel):
    """Code output"""

    imports: str = Field(description="The import statements for the code")
    code: str = Field(description="The main code block")
    summary: str = Field(
        description="A brief summary of previous codes and the flow they follow"
    )
    # variables: Dict[str, str] = Field(
    #     description="A dictionary of variable names and their descriptions"
    # )
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


class CodeGenerationAgent:
    def __init__(
        self,
        config,
        proxy,
        nb_executor: NotebookExecutorInterface,
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
        self.uploaded=False
        # IMPROVED_CODE_GEN_PROMPT.messages[0]

        # zeroshot_chain =  LangChainPredict(self.code_gen_prompt, self.llm) |  self.output_parser
        # zeroshot_chain = LangChainModule(zeroshot_chain)  #
        # self.code_gen_chain = zeroshot_chain
        self.code_gen_chain = self.code_gen_prompt | self.llm

        self.workflow = self.create_workflow()

    def generate(self, state: CodeGraphState):
        logger.info("---GENERATING CODE SOLUTION---")
        kaggle_state = state["kaggle_state"]
        iterations = state["iterations"]
        # error = state["error"]
        messages = state["messages"]

        try:
            code_solution: GeneratedCode = self.code_gen_chain.invoke(
                {
                    "problem_description": kaggle_state.problem_description,
                    "modelInfo": kaggle_state.modelInfo,
                    "planned_tasks": kaggle_state.planned_tasks,
                    "evaluation_metric": kaggle_state.evaluation_metric,
                    "format_instructions": self.format_instructions,
                    "messages": messages,
                    "current_task": str(kaggle_state.enhanced_tasks[kaggle_state.index]),
                    "format_instructions": self.format_instructions,
                    "previous_tasks": kaggle_state.get_task_results(),
                },
                config=self.config,
            )
            m = [
                (
                    "ai",
                    f"""
codes description:
{code_solution.description}

code is :
{code_solution.imports}

{code_solution.code}

"""
                )
            ]

            logger.info("Code generation successful.")
            return {
                "generation": code_solution,
                "iterations": iterations,
                "error": "no",
                "messages": m,
            }
        except Exception as e:
            logger.error(f"Error during code generation: {e}", exc_info=True)
            return {
                "generation": GeneratedCode(imports="", code="", description=""),
                "iterations": iterations,
                "error": "yes",
                "erro_msg": str(e),
                "iteration": iterations + 1,
                "messages": messages,
            }

    def check_and_correct_variables(self, state: CodeGraphState):
        code_solution = state["generation"]
        ast_variables = state["ast_variables"]
        llm_variables = set(code_solution.variables.keys())

        if ast_variables != llm_variables:
            missing_in_llm = ast_variables - llm_variables
            extra_in_llm = llm_variables - ast_variables

            corrected_variables = self.llm.invoke(
                self.variable_correction_prompt.format(
                    code=code_solution.code,
                    current_variables=code_solution.variables,
                    missing_variables=list(missing_in_llm),
                    extra_variables=list(extra_in_llm),
                )
            )

            code_solution.variables = corrected_variables

        return {"generation": code_solution}

    def extract_variables_from_ast(self, state: CodeGraphState) -> dict[str, set[str]]:
        code = state["generation"].code
        tree = ast.parse(code)
        variables = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                variables.add(node.id)
            elif isinstance(node, ast.arg):
                variables.add(node.arg)

        return {"ast_variables": variables}

    def code_check(self, state: CodeGraphState):
        logger.info("---CHECKING CODE---")
        code_solution = state["generation"]
        iterations = state["iterations"]

        imports = code_solution.imports
        code = code_solution.code

        try:  #
            result = self.nb_executor.test_and_execute(imports + "\n" + code)
            result = exec2s(result)
            logger.info("✅ ---NO CODE TEST FAILURES--- ✅")
            return {
                "generation": code_solution,
                "iterations": iterations,
                "error": "no",
                "result": result,
            }
        except CellError as e:

            logger.warn(f"❌ ---CODE CHECK FAILED: {e.ename}--- ❌")

            m = [
                (
                    "user",
                    f"""
Your latest solution to code failed the code execution test: 
explain what error it is and how to solve it
**error_message:**
```
Error name {remove_color(cc(str(e.traceback)))}
- 
```
""",
                )
            ]
            llm = ChatOpenAI(
                base_url=self.base_url,
                model="gpt-4o-mini",
                http_client=self.proxy,
                temperature=0,
            )
            kaggle_state = state["kaggle_state"]
            res = (self.code_gen_prompt | llm | StrOutputParser()).invoke(
                {
                    "problem_description": kaggle_state.problem_description,
                    "modelInfo": kaggle_state.modelInfo,
                    "planned_tasks": kaggle_state.planned_tasks,
                    "evaluation_metric": kaggle_state.evaluation_metric,
                    "format_instructions": self.format_instructions,
                    "messages": state["messages"] + m,
                    "current_task": str(
                        kaggle_state.enhanced_tasks[kaggle_state.index]
                    ),
                    "previous_tasks": kaggle_state.get_task_results(),
                },
                config=self.config,
            )
            m += [("ai", res)]

            return {
                "generation": code_solution,
                "iterations": iterations,
                "error": "yes",
                "erro_msg": e.traceback,
                "iteration": state["iterations"] + 1,
                "messages": m,
            }

    def decide_to_finish(self, state: CodeGraphState):
        error = state["error"]
        iterations = state["iterations"]

        if error == "no" or iterations == self.max_iterations:
            logger.info("---DECISION: FINISH---")
            return "end"
        elif error == "yes":
            return "reset_procedure"
        else:
            logger.warn("---DECISION: RE-TRY SOLUTION---", "iters No.", iterations)
            return "generate"

    def create_workflow(self):
        workflow = StateGraph(CodeGraphState)

        workflow.add_node("generate", self.generate)
        workflow.add_node("reset_procedure", self.__reset_procedure)
        # workflow.add_node("extract_var_ast", self.extract_variables_from_ast)
        # workflow.add_node(
        #     "check_and_correct_variables", self.check_and_correct_variables
        # )
        # ast_variables = self.extract_variables_from_ast(code_solution.code)
        # code_solution = self.check_and_correct_variables(code_solution)

        workflow.add_node("check_code", self.code_check)

        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", "check_code")
        workflow.add_edge("reset_procedure", "generate")
        # workflow.add_edge("extract_var_ast", "check_and_correct_variables")
        # workflow.add_edge("check_and_correct_variables", "check_code")
        workflow.add_conditional_edges(
            "check_code",
            self.decide_to_finish,
            {"end": END, "generate": "generate", "reset_procedure": "reset_procedure"},
        )

        return workflow.compile()

    def reflect(self, state: CodeGraphState):
        """
        Reflect on errors

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation
        """

        logger.info("---REFLECTING ON ERRORS---")

        # State
        messages = state["messages"]
        iterations = state["iterations"]
        code_solution = state["generation"]

        # Prompt reflection

        # Add reflection
        kaggle_state = state["kaggle_state"]

        try:
            reflections = self.code_gen_chain.invoke(
                {
                    "problem_description": kaggle_state.problem_description,
                    "modelInfo": kaggle_state.modelInfo,
                    "planned_tasks": kaggle_state.planned_tasks,
                    "evaluation_metric": kaggle_state.evaluation_metric,
                    "messages": state["messages"],
                    "current_task": str(kaggle_state.enhanced_tasks[kaggle_state.index]),
                    "format_instructions": self.format_instructions,
                    "previous_tasks": kaggle_state.get_task_results(),
                },
            )
            messages += [("assistant", f"Here are reflections on the error: {reflections}")]
            logger.info("Reflection on errors completed successfully.")
            return {
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
            }
        except Exception as e:
            logger.error(f"Error during reflection: {e}", exc_info=True)
            return state  # Return current state if reflection fails

    def __reset_procedure(self, state: CodeGraphState):
        kaggle_state = state["kaggle_state"]
        task_codes_results = kaggle_state.task_codes_results

        new_task_codes = []
        self.nb_executor.reset()
        for t, c, r in task_codes_results:
            try:
                new_result = self.nb_executor.test_and_execute(str(c))
                new_task_codes.append((t, c, exec2s(new_result)))
            except Exception as e:
                raise e

        kaggle_state.task_codes_results = new_task_codes
        self.nb_executor.is_restarted = False
        return {"error": "no", "kaggle_state": kaggle_state}
    def _cp_files(self, state: CodeGraphState):
        # self.dataset_path = "./ongoing/train.csv"
        # self.test_dataset_path = "./ongoing/test.csv"

        with open(state.dataset_path,'rb') as f:
            env_var = self.nb_executor.upload_file_env(f)
        with open(state.test_dataset_path,'rb') as f:
            env_var = self.nb_executor.upload_file_env(f)
        
    def __call__(self, state: KaggleProblemState):
        initial_state = {
            "kaggle_state": state,
            "iterations": 0,
            "generation": GeneratedCode(imports="", code="", variables={}, summary=""),
            "messages": [],
        }
        task_codes = state.task_codes_results
        if self.uploaded==False:
            self._cp_files(state)
            self.uploaded=True
        result = self.workflow.invoke(initial_state, config=self.config)

        # print( str(result.get("result", "nothing")))
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
            "summary": result["generation"].summary,
            # "variables": result["generation"].variables,
        }
