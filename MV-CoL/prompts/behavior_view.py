"""View B: dataset-configured, local behavior-cue MV-CoL prompt."""

from prompts.common import PromptSpec

VIEW_ID = "B"
VIEW_NAME = "behavior_cue"

def build_messages(text: str, spec: PromptSpec) -> list[dict[str, str]]:
    system_prompt = (
        f"Task dataset: {spec.dataset_name}\n"
        f"Task description: {spec.task_description}\n"
        f"Instruction: {spec.task_instruction}\n\n"
        "Focus on local, text-supported behavior cues:\n"
        f"{spec.behavior_lines()}\n\n"
        "Resolve ambiguous cues with these configured label definitions:\n"
        f"{spec.definition_lines()}\n\n"
        f"Return exactly one output label from: {spec.output_constraint()}."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Input text:\n{text}\n\nBehavior-cue label:"},
    ]
