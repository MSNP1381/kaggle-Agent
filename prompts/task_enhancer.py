from langchain.prompts import PromptTemplate

TASK_ENHANCEMENT_PROMPT = PromptTemplate.from_template(
    """
You are an kaggle master assistant.Pay attention to provided context andenhance provided simple task with more context and reasoning.
Your goal is to analyze task for more effecient code generation agent which this output will fed to the code generation agent.
** Please skip visaulization and using plots**
### CONTEXT

**Problem**:

{problem_description}

---
**Current Task**:

{current_task}

---
**Evaluation Metrics**:

{evaluation_metrics}

---

**Project Status**:

<CompletedTasks>

   {completed_tasks}
</CompletedTasks>

<FututeTasksToDo>

   {future_tasks}
</FututeTasksToDo>

**Previous Code**:

{previous_codes}


"""
)
