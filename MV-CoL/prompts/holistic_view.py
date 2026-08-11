"""View C: dataset-configured, holistic-judgment MV-CoL prompt."""

from prompts.common import PromptSpec

VIEW_ID = "C"
VIEW_NAME = "holistic_judgment"

def build_messages(text: str, spec: PromptSpec) -> list[dict[str, str]]:
    system_prompt = (
        f"Task dataset: {spec.dataset_name}\n"
        f"Task description: {spec.task_description}\n"
        f"Instruction: {spec.task_instruction}\n\n"
        "Judge the input as a whole. Consider its complete communicative intent "
        "and overall meaning, using the configured criteria rather than isolated "
        "keywords alone.\n\n"
        "Label definitions:\n"
        f"{spec.definition_lines()}\n\n"
        "Behavior cues are supporting evidence, not a keyword-only rule:\n"
        f"{spec.behavior_lines()}\n\n"
        f"Return exactly one output label from: {spec.output_constraint()}."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Complete input:\n{text}\n\nHolistic label:"},
    ]
