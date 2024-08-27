import ast
import json
import operator
import pprint
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from typing import Annotated, List, TypedDict, Set, Dict
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.pydantic_v1 import BaseModel, Field
from nbclient.exceptions import CellExecutionError
from states.code import Code
from states.main import KaggleProblemState
from nbexecutor import NBExecutor
from prompts.code_generation_prompt import (
    IMPROVED_CODE_GEN_PROMPT,
    VARIABLE_CORRECTION_PROMPT,
)

import re

from utils import cc


def remove_color(text):
    ansi_escape = re.compile(
        r"\x1B\[[0-?]*[ -/]*[@-~]"
    )  # I looked it up and the library uses ANSI codes internally so I believe this is the right re.compile
    return ansi_escape.sub("", text)


from typing import Dict


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


class CodeGraphState(TypedDict):
    error: str
    generation: GeneratedCode
    iterations: int
    kaggle_state: KaggleProblemState
    ast_variables: set[str]
    messages: Annotated[List, operator.add]


class CodeGenerationAgent:
    def __init__(
        self,
        config,
        proxy,
        nb_executor,
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
        # IMPROVED_CODE_GEN_PROMPT.messages[0]

        # zeroshot_chain =  LangChainPredict(self.code_gen_prompt, self.llm) |  self.output_parser
        # zeroshot_chain = LangChainModule(zeroshot_chain)  #
        # self.code_gen_chain = zeroshot_chain
        self.code_gen_chain = self.code_gen_prompt | self.llm

        self.workflow = self.create_workflow()
        s=self.workflow.get_graph().draw_mermaid()        
        with open("code_mermaid.txt",'w')as f:
            f.write(s)
        print(s)

    def generate(self, state: CodeGraphState):
        print("---GENERATING CODE SOLUTION---")
        kaggle_state = state["kaggle_state"]
        iterations = state["iterations"]
        error = state["error"]
        messages = state["messages"]

        code_solution: GeneratedCode = self.code_gen_chain.invoke(
            {
                "problem_description": kaggle_state.problem_description,
                "model_info": kaggle_state.model_info,
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


""",
            )
        ]

        # Check and correct variables

        iterations += 1
        return {
            "generation": code_solution,
            "iterations": iterations,
            "error": "no",
            "messages": m,
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
        print("---CHECKING CODE---")
        code_solution = state["generation"]
        iterations = state["iterations"]

        imports = code_solution.imports
        code = code_solution.code

        try:
            result = self.nb_executor.test_and_execute(imports + "\n" + code)
            print("---NO CODE TEST FAILURES---")
            return {
                "generation": code_solution,
                "iterations": iterations,
                "error": "no",
                "result": result,
            }
        except Exception as e:
            print(f"---CODE CHECK FAILED: {e.ename}---")

            m = [
                (
                    "user",
                    f"""\
Your latest solution to code failed the code execution test: 
explain what error it is and how to solve it
**error_message:**
```
   Error name {remove_color(cc(e.traceback))}
- 
```
""",
                )
            ]
            llm = ChatOpenAI(
                base_url=self.base_url,
                model="gpt-4o",
                http_client=self.proxy,
                temperature=0,
            )
            kaggle_state = state["kaggle_state"]
            res = (self.code_gen_prompt | llm | StrOutputParser()).invoke(
                {
                    "problem_description": kaggle_state.problem_description,
                    "model_info": kaggle_state.model_info,
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
                "messages": m,
            }

    def decide_to_finish(self, state: CodeGraphState):
        error = state["error"]
        iterations = state["iterations"]

        if error == "no" or iterations == self.max_iterations:
            print("---DECISION: FINISH---")
            return "end"
        else:
            print("---DECISION: RE-TRY SOLUTION---", "iters No.", iterations)
            return "generate"

    def create_workflow(self):
        workflow = StateGraph(CodeGraphState)

        workflow.add_node("generate", self.generate)
        # workflow.add_node("extract_var_ast", self.extract_variables_from_ast)
        # workflow.add_node(
        #     "check_and_correct_variables", self.check_and_correct_variables
        # )
        # ast_variables = self.extract_variables_from_ast(code_solution.code)
        # code_solution = self.check_and_correct_variables(code_solution)

        workflow.add_node("check_code", self.code_check)

        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", "check_code")
        # workflow.add_edge("extract_var_ast", "check_and_correct_variables")
        # workflow.add_edge("check_and_correct_variables", "check_code")
        workflow.add_conditional_edges(
            "check_code",
            self.decide_to_finish,
            {
                "end": END,
                "generate": "generate",
            },
        )

        return workflow.compile(debug=True)

    def reflect(self, state: CodeGraphState):
        """
        Reflect on errors

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation
        """

        print("---GENERATING CODE SOLUTION---")

        # State
        messages = state["messages"]
        iterations = state["iterations"]
        code_solution = state["generation"]

        # Prompt reflection

        # Add reflection
        kaggle_state = state["kaggle_state"]

        reflections = self.code_gen_chain.invoke(
            {
                "problem_description": kaggle_state.problem_description,
                "model_info": kaggle_state.model_info,
                "planned_tasks": kaggle_state.planned_tasks,
                "evaluation_metric": kaggle_state.evaluation_metric,
                "messages": state["messages"],
                "current_task": str(kaggle_state.enhanced_tasks[kaggle_state.index]),
                "format_instructions": self.format_instructions,
                "previous_tasks": kaggle_state.get_task_results(),
            },
        )
        messages += [("assistant", f"Here are reflections on the error: {reflections}")]
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
        }

    def __call__(self, state: KaggleProblemState):
        initial_state = {
            "kaggle_state": state,
            "iterations": 0,
            "generation": GeneratedCode(imports="", code="", variables={}, summary=""),
            "messages": [],
        }
        task_codes = state.task_codes_results

        result = self.workflow.invoke(initial_state, config=self.config)
        task_codes.append(
            (
                state.enhanced_tasks[state.index],
                Code(
                    imports=result["generation"].imports,
                    code=result["generation"].code,
                    description=result["generation"].description,
                ),
                result.get("result", ""),
            )
        )
        return {
            "task_codes_results": task_codes,
            "summary": result["generation"].summary,
            # "variables": result["generation"].variables,
        }
