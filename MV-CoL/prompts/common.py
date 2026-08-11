"""Dataset-configured prompt metadata shared by all three MV-CoL views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptSpec:
    """Validated task information used to render dataset-specific prompts."""

    dataset_name: str
    task_description: str
    task_instruction: str
    labels: tuple[str, ...]
    label_definitions: dict[str, str]
    behavior_cues: dict[str, tuple[str, ...]]
    output_labels: tuple[str, ...]

    @classmethod
    def from_dataset_config(cls, data_config: dict[str, Any]) -> "PromptSpec":
        labels = tuple(str(value) for value in data_config["labels"])
        output_labels = tuple(str(value) for value in data_config["output_labels"])
        definitions = {
            str(label): str(definition).strip()
            for label, definition in data_config["label_definitions"].items()
        }
        cues: dict[str, tuple[str, ...]] = {}
        for label, values in data_config["behavior_cues"].items():
            if isinstance(values, str):
                values = [values]
            cues[str(label)] = tuple(str(value).strip() for value in values)
        return cls(
            dataset_name=str(data_config["dataset_name"]).strip(),
            task_description=str(data_config["task_description"]).strip(),
            task_instruction=str(data_config["task_instruction"]).strip(),
            labels=labels,
            label_definitions=definitions,
            behavior_cues=cues,
            output_labels=output_labels,
        )

    def definition_lines(self) -> str:
        return "\n".join(
            f"- {label}: {self.label_definitions[label]}" for label in self.labels
        )

    def behavior_lines(self) -> str:
        lines: list[str] = []
        for label in self.labels:
            joined_cues = "; ".join(self.behavior_cues[label])
            lines.append(f"- {label}: {joined_cues}")
        return "\n".join(lines)

    def output_constraint(self) -> str:
        return ", ".join(self.output_labels)
