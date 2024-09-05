CHALLENGE_DESCRIPTOIN_PROMPT = [
    (
        "system",
        "You are a helpful AI assistant. Your job is to summarize the challenge description of a Kaggle-ML competition. Provide a clear and concise summary of the challenge, including its objectives and any important details. Explain your reasoning step-by-step.",
    ),
    (
        "user",
        "challenge description is :\n\n DESCRIPTION:```markdown\n{text}\n```",
    ),
]

CHALLENGE_EVALUATION_PROMPT = [
    (
        "system",
        "You are a helpful AI assistant. Your job is to explain the evaluation criteria of a Kaggle-ML competition. Provide a detailed explanation of how the submissions will be evaluated, including any metrics or methods used. Explain your reasoning step-by-step.",
    ),
    (
        "user",
        "use this formatting instruction: {format_instructions} \n challenge evaluation decription is :\n\n ```markdown\n{text}\n```",
    ),
]
CHALLENGE_DATA_PROMPT = [
    (
        "system",
        "You are a helpful AI assistant. Your job is to describe the data provided for a Kaggle-ML competition. Provide a detailed description of the dataset, including its structure, features, and any important details. Explain your reasoning step-by-step. what is the target column and if there isn't a taget column how to get evaluation in dataset.",
    ),
    ("user", "use this formatting instruction: {format_instructions}\nchallenge data decription is :\n\n ```markdown\n{text}\n```"),
]
