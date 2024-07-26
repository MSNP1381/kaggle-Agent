import os
from pathlib import Path
import orjson
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from typing import List, TypedDict
from langchain.output_parsers import PydanticOutputParser
from states.enhancer import EnhancedTask
from states.main import KaggleProblemState
from nbexecutor import NBExecutor
from prompts.code_generatino_prompt import SIMPLIFIED_CODE_GEN_PROMPT
from states.main import Code

# from utils import encode_json


def encode_json(obj):
    if isinstance(obj, Code):
        return str(obj)


class GraphState(TypedDict):
    error: str
    messages: List
    generation: Code
    iterations: int
    context: str
    enhanced_task: EnhancedTask


class CodeGenerationAgent:
    def __init__(
        self,
        config,
        proxy,
        nb_executor: NBExecutor,
        model="gpt-4o-mini",
        max_iterations=3,
    ):
        self.max_iterations = max_iterations
        self.config = config
        self.nb_executor = nb_executor
        self.llm = ChatOpenAI(model=model, http_client=proxy, temperature=0)
        self.code_gen_prompt = SIMPLIFIED_CODE_GEN_PROMPT
        self.output_parser = PydanticOutputParser(pydantic_object=Code)
        self.format_instructions = self.output_parser.get_format_instructions()
        self.code_gen_chain = self.code_gen_prompt | self.llm | self.output_parser
        self.workflow = self.create_workflow()

    def generate(self, state: GraphState):
        print("---GENERATING CODE SOLUTION---")
        messages = state["messages"]
        iterations = state["iterations"]
        error = state["error"]
        enhanced_task = state["enhanced_task"]

        if error == "yes":
            messages.append(
                (
                    "human",
                    "The previous solution had errors. Please try again, ensuring all imports are correct and the "
                    "code is executable.",
                )
            )

        human_message = messages  # Get the latest human message

        # Prepare task requirements string
        # task_requirements = "\n".join(enhanced_task.get("requirements", []))

        code_solution = self.code_gen_chain.invoke(
            {
                **state["context"],
                "conversation": human_message,
                # "format_instructions": self.format_instructions,
            },
            config=self.config,
        )
        # Ensure all fields are present in the Code object
        code_solution_dict = code_solution.dict()
        for field in ["imports", "code", "description"]:
            if field not in code_solution_dict or code_solution_dict[field] is None:
                code_solution_dict[field] = ""  # Set to empty string if missing

        code_solution = Code(**code_solution_dict)

        messages.append(
            (
                "assistant",
                f"Imports:\n{code_solution.imports}\nCode:\n{code_solution.code}\nDescription:\n{code_solution.description}",
            )
        )

        iterations += 1
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
        }

    def code_check(self, state: GraphState):
        print("---CHECKING CODE---")
        messages = state["messages"]
        code_solution = state["generation"]
        iterations = state["iterations"]

        imports = code_solution.imports
        code = code_solution.code

        try:
            self.nb_executor.test_and_execute(imports)
            self.nb_executor.test_and_execute(imports + "\n" + code)

            print("---NO CODE TEST FAILURES---")
            return {
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": "no",
            }
        except Exception as e:
            print(f"---CODE CHECK FAILED: {str(e)}---")
            error_message = [
                ("human", f"Your solution failed with the following error: {str(e)}")
            ]
            messages += error_message
            return {
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": "yes",
            }

    def decide_to_finish(self, state: GraphState):
        error = state["error"]
        iterations = state["iterations"]

        if error == "no" or iterations == self.max_iterations:
            print("---DECISION: FINISH---")
            return "end"
        else:
            print("---DECISION: RE-TRY SOLUTION---")
            return "generate"

    def create_workflow(self):
        workflow = StateGraph(GraphState)

        # workflow.add_node("start_generate", self.generate_code)
        workflow.add_node("generate", self.generate)
        workflow.add_node("check_code", self.code_check)
        # workflow.add_node("conv_result", self._conv)

        # workflow.add_edge(START, "start_generate")
        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", "check_code")
        workflow.add_conditional_edges(
            "check_code",
            self.decide_to_finish,
            {
                "end": END,
                "generate": "generate",
            },
        )
        # workflow.add_edge("conv_result", END)

        return workflow.compile()

    # @staticmethod
    def kaggle_conv_code(self, state: KaggleProblemState):
        # task_codes_results
        # human message [x]
        task_code_pairs_formatted = []
        index = 1
        for i, j in state.task_codes_results.items():
            (enh_task, code, res) = j
            task_code_pairs_formatted += [
                (
                    "human",
                    f"""
*************
task No.{index}

Task:

{i}

---------------------
Requirements:
{"\n".join(enh_task.requirements) }

""",
                ),
                (
                    "assistant",
                    f"""Code :

{code}

---------------------

Result:

{res}
*************""",
                ),
            ]

            index += 1

        # for i in state.task_codes_results

        task_code_pairs_formatted += [
            (
                "human",
                f"Current Task :\n{state.enhanced_task.task}\n\nTask Requirements:\n{state.enhanced_task.requirements}\n\nAdhere strictly to the following output format: \n {self.format_instructions}",
            )
        ]

        context = {
            "problem_description": state.problem_description,
            "project_state": {
                "dataset_info": state.dataset_info,
                # "current_task": state.enhanced_task.task,
                "model_info": state.model_info,
                "planned_tasks": state.planned_tasks,
                "evaluation_metric": state.evaluation_metric,
                "best_score": state.best_score,
            },
        }

        initial_state = {
            "messages": task_code_pairs_formatted,
            "iterations": 0,
            "context": context,
            "error": "no",
            "enhanced_task": state.enhanced_task.dict(),  # Convert EnhancedTask to dict
            "generation": Code(imports="", code="", description=""),
        }
        Path("log.txt").touch()
        with open("log.txt", "a") as f:

            f.write(
                str(orjson.dumps(initial_state, default=encode_json), encoding="utf8")
            )
            f.write("\n\n************************************************\n\n\n")
        result = self.workflow.invoke(initial_state, config=self.config)
        return {
            "task_codes_results": {
                state.enhanced_task.task: (
                    state.enhanced_task,
                    result["generation"],
                    "",
                )
            }
        }

    def _conv(self, state):
        # code = self.generate_code(state)
        # state.task_codes_results[state.enhanced_task.task] = (code, '')
        print("****--" * 10)
        print({"task_codes_results": {self.task.task: (state["generation"], "")}})

        return {"task_codes_results": {self.task.task: (state["generation"], "")}}

    def __call__(self, state: KaggleProblemState):

        return self.kaggle_conv_code(state)

    #     state.task_codes_results[state.enhanced_task.task] = (code, "")

    #     return {"task_codes_results": state.task_codes_results}

    # return self.generate_code(state)
