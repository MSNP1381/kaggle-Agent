import json
import re
from abc import ABC
from dataclasses import asdict, dataclass
from typing import List, Union

import black
from langchain_core.documents import Document

# from states.main import KaggleProblemState
from states.write_challenge_docs import TEXT


@dataclass
class CellOutput:
    output_type: str
    name: str
    text: str

    def to_json(self) -> str:
        # Convert the dataclass to a dictionary and then to a JSON string
        return json.dumps(asdict(self))


class CellError(Exception):
    def __init__(self, ename, evalue, traceback):
        self.ename = ename
        self.evalue = evalue
        self.traceback = traceback

    def __repr__(self) -> str:
        return f"error: {self.ename}"

    # @property
    # def ename(self)->str:
    #     return self.name
    # @property
    # def evalue(self)->str:
    #     return self.value
    # @property
    # def traceback(self)->str:
    #     return self.traceback_


class CellResult:
    def __init__(self, result):
        # super().__init__(result_instance.is_main_result,result_instance.extra)
        self.result = result

    @property
    def output_str(self) -> str:
        self.result

    def __str__(self) -> str:
        return self.result

    # def __repr__(self) -> str:
    #     return self.result


class NotebookExecutorInterface(ABC):
    def create_nb(self) -> str:
        pass

    def upload_file_env(self):
        pass

    def __init__(self, execution_instance) -> None:
        self.executor = None
        self.is_restarted = False

    def test_and_execute(self, new_code: str) -> CellResult:
        pass

    def reset(self) -> None:
        pass


class NotebookFailError(Exception):
    """Exception raised for errors in the notebook execution process."""

    def __init__(self, message: str, code: str = None):
        self.message = message
        self.code = code
        super().__init__(self.message)

    def __str__(self):
        if self.code:
            return f"{self.message} (Code: {self.code})"
        return self.message


def dict_concat(a, b):
    return {**a, **b}


def cc(s: str):
    return s.replace("\\n", "\n")


def exec2s(data: Union[CellResult, List[CellResult]]) -> str:
    out_s = ""
    if isinstance(data, List):
        out_s = "\n---\n".join(map(str, data))
        out_s = out_s.replace("\\n", "\n")
    else:
        out_s = str(data)

    return out_s


def get_top_10_percent_mean(json_path: str) -> float:
    """
    Calculate the mean of the top 10 percent score values from the provided JSON schema.

    :param json_path: A string containing the path to the JSON file with score values.
    :return: The mean of the top 10 percent score values.
    """

    with open(json_path, "r") as file:
        json_data = json.load(file)

    submissions = json_data.get("submissions", [])
    if not submissions:
        return 0.0

    scores = []
    for submission in submissions:
        score_str = submission.get("score")
        if score_str is not None:
            try:
                score = float(score_str)
                scores.append(score)
            except ValueError:
                continue

    if not scores:
        return 0.0

    scores.sort(reverse=True)
    top_10_percent_count = max(1, len(scores) // 10)
    top_10_percent_scores = scores[:top_10_percent_count]

    return sum(top_10_percent_scores) / len(top_10_percent_scores)


def append_url(base_url: str, sub_url: str, use_https: bool = True) -> str:
    """
    Append a sub URL to a base URL and add the appropriate protocol (http or https).

    :param base_url: The base URL to which the sub URL will be appended.
    :param sub_url: The sub URL to append to the base URL.
    :param use_https: Boolean flag to determine whether to use https (default) or http.
    :return: The complete URL with the appropriate protocol.
    """
    protocol = "https://" if use_https else "http://"
    if not base_url.startswith("http://") and not base_url.startswith("https://"):
        base_url = protocol + base_url
    if not base_url.endswith("/"):
        base_url += "/"
    return base_url + sub_url


def state2doc_write(state) -> str:
    name = state.challenge_url.split("/")[-2]
    important_notes = open("./important_notes/important_notes.txt").read()
    d = {
        "challenge_name": name,
        "problem_description": state.problem_description,
        "quantitative_analysis": state.quantitative_analysis,
        "qualitative_analysis": state.qualitative_analysis,
        "feature_recommendations": state.feature_recommendations,
        "dataset_info": state.dataset_info,
        "evaluation": state.evaluation_metric,
        "important_notes": important_notes,
    }

    s = TEXT.format(**d)
    open("./input/doc.txt", "w").write(s)
    with open("./input/doc_dict.json", "w") as file:
        json.dump(d, file)
    return TEXT.format(**d)


def state2retrieve_doc():
    with open("./input/doc_dict.json") as file:
        d = json.load(file)
    data = []
    for k, v in d.items():
        data.append(Document(page_content=v, metadata={"source": k}))
    return data


def format_code(code) -> str:
    """Format Python code using Black."""
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        return code


def is_valid_python_script(script):
    """Check if a script is a valid Python script."""
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def extract_code(text):
    """Extract python code blocks from the text."""
    parsed_codes = []

    # When code is in a text or python block
    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)

    # When the entire text is code or backticks of the code block is missing
    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    # validate the parsed codes
    valid_code_blocks = [
        format_code(c) for c in parsed_codes if is_valid_python_script(c)
    ]
    return format_code("\n\n".join(valid_code_blocks))


def extract_text_up_to_code(s):
    """Extract (presumed) natural language text up to the start of the first code block."""
    if "```" not in s:
        return ""
    return s[: s.find("```")].strip()


def extract_markdown(text):
    """Extract markdown content from the text."""
    parsed_markdown = []

    # Extract content between markdown code blocks
    matches = re.split(r"```[\s\S]*?```", text)

    for match in matches:
        # Remove any remaining backticks
        cleaned_match = re.sub(r"`", "", match)
        # Remove any remaining Python prompts
        cleaned_match = re.sub(r"^>>>\s?", "", cleaned_match, flags=re.MULTILINE)
        # Remove leading/trailing whitespace
        cleaned_match = cleaned_match.strip()

        if cleaned_match:
            parsed_markdown.append(cleaned_match)

    return "\n\n".join(parsed_markdown)
