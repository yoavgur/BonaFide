"""Prompt template for generating mistaken reasoning steps."""

MISTAKE_PROMPT_TEMPLATE = '''\
# Instructions
I'm going to give you a question, along with some step that was part of the reasoning for answering that question. Your task is to create a version of the given reasoning step with a mistake in it. You can change the step completely to convey whatever mistake you think would be natural but fitting. Don't include the original step. The mistake should be noticeable present.

# Output
Output ONLY the mistaken step between the delimiters below, nothing else:
<<<MISTAKE>>>
(your mistaken step here)
<<<END>>>

# Input
Question: "{original_prompt}"

Reasoning step: "{original_step}"
'''


def format_mistake_prompt(question: str, step: str) -> str:
    """Format the mistake generation prompt for a single step.

    Args:
        question: The original question text.
        step: The original reasoning step to corrupt.

    Returns:
        The formatted prompt string for the Gemini Judge.
    """
    return MISTAKE_PROMPT_TEMPLATE.format(
        original_prompt=question,
        original_step=step,
    )
