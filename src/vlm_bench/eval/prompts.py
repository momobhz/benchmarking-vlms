from __future__ import annotations

PROMPT_MODE_CHOICES = ("cot", "zero_shot")

COT_INSTRUCTION = (
    "Answer the following multiple choice question by selecting the letter "
    "(A, B, C, D, or E). Reason step by step about the answer, and show your "
    "work for each step. Only after that, proceed to the final answer. Please "
    "answer the question and provide the correct option letter, e.g., A, B, C, "
    "D, E, at the end."
)

ZERO_SHOT_INSTRUCTION = (
    "Answer the following multiple choice question by selecting the letter "
    "(A, B, C, D, or E). Only output the correct option letter, i.e., A, B, C, "
    "D, or E."
)


def build_prompt_instruction(prompt_mode: str) -> str:
    if prompt_mode == "cot":
        return COT_INSTRUCTION
    if prompt_mode == "zero_shot":
        return ZERO_SHOT_INSTRUCTION
    raise ValueError(
        f"Unsupported prompt mode: {prompt_mode!r}. "
        f"Expected one of: {', '.join(PROMPT_MODE_CHOICES)}"
    )


def apply_prompt_mode(questions: list[str], prompt_mode: str) -> list[str]:
    instruction = build_prompt_instruction(prompt_mode)
    return [f"{instruction} {question}" for question in questions]
