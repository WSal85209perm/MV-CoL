"""View A: dataset-configured, definition-guided MV-CoL prompt."""

from prompts.common import PromptSpec

VIEW_ID = "A"
VIEW_NAME = "definition_guided"

def build_messages(text: str, spec: PromptSpec) -> list[dict[str, str]]:
    system_prompt = (
        f"Task dataset: {spec.dataset_name}\n"
        f"Task description: {spec.task_description}\n"
        f"Instruction: {spec.task_instruction}\n\n"
        "Use these label definitions and decision criteria:\n"
        f"{spec.definition_lines()}\n\n"
        "Configured behavioral evidence:\n"
        f"{spec.behavior_lines()}\n\n"
        f"Return exactly one output label from: {spec.output_constraint()}."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Input text:\n{text}\n\nLabel:"},
    ]
