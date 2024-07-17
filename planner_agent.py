import httpx
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from code_generation_agent import CodeGenerationAgent, Code

class KaggleProblemPlanner:
    def __init__(self, config,proxy):
        # proxy = httpx.Client(proxy="http://127.0.0.1:2081")
        self.config = config
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", http_client=proxy,temperature=0)
        self.code_generation_agent = CodeGenerationAgent(config=config,proxy=proxy)
        
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
        3. Ensure the plan covers all necessary steps:  preprocessing, feature engineering, model selection, training, and evaluation.
        
        Notes:
        ensure dont call plt.show() method for plots in codes
        don't use any visually formated like rendred Html outputs just use text or dict for outputs 

        Respond with a list of planned tasks, each on a new line.
        """)

    def plan(self, state):
        response = self.llm.invoke(self.planner_prompt.format_messages(
            problem_description=state.problem_description,
            state=str(state.__dict__)
        ), config=self.config)
        return response.content.strip().split("\n")

    def generate_code(self, state) -> Code:
        # Prepare the context for the CodeGenerationAgent
        context = f"""
        Problem Description:
        {state.problem_description}

        Current Task: {state.current_task}

        Project State:
        {str(state.__dict__)}
        """
        
        # Use the CodeGenerationAgent to generate code
        return self.code_generation_agent.generate_code(context, state.current_task)

# Usage example
if __name__ == "__main__":
    class State:
        def __init__(self, problem_description, current_task):
            self.problem_description = problem_description
            self.current_task = current_task

    config = {}  # Add any necessary configuration
    planner = KaggleProblemPlanner(config)
    
    state = State(
        problem_description="Classify images of cats and dogs",
        current_task="Create a function to load and preprocess image data"
    )
    
    plan = planner.plan(state)
    print("Plan:", plan)
    
    code_solution = planner.generate_code(state)
    print(f"Generated Code:\nImports:\n{code_solution.imports}\n\nCode:\n{code_solution.code}\n\nDescription:\n{code_solution.description}")