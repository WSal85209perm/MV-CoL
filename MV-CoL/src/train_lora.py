"""Four-bit LoRA training with MV-CoL's joint three-view objective."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from prompts.common import PromptSpec
from prompts.behavior_view import build_messages as behavior_messages
from prompts.definition_view import build_messages as definition_messages
from prompts.holistic_view import build_messages as holistic_messages
from src.losses import PerViewJointLoss, ProjectionHead


VIEW_BUILDERS: tuple[Callable[[str, PromptSpec], list[dict[str, str]]], ...] = (
    definition_messages,
    behavior_messages,
    holistic_messages,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str | Path, data_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Read and validate a user-supplied split without modifying its membership."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    text_key = data_cfg["text_field"]
    label_key = data_cfg["label_field"]
    id_key = data_cfg.get("id_field", "id")
    allowed = set(data_cfg["labels"])
    label_aliases = {
        str(raw_value): str(label)
        for raw_value, label in data_cfg.get("label_aliases", {}).items()
    }
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if text_key not in row or label_key not in row:
                raise ValueError(f"{path}:{line_number} lacks '{text_key}' or '{label_key}'")
            raw_label = str(row[label_key])
            label = label_aliases.get(raw_label, raw_label)
            if label not in allowed:
                raise ValueError(
                    f"{path}:{line_number} has unknown label {row[label_key]!r}; "
                    "configure data.label_aliases when source values differ from label names"
                )
            rows.append(
                {
                    "id": str(row.get(id_key, f"{path.stem}-{line_number}")),
                    "text": str(row[text_key]),
                    "label": label,
                }
            )
    if not rows:
        raise ValueError(f"Dataset split is empty: {path}")
    return rows


def render_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Render chat messages with the model template, with a portable fallback."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return "\n\n".join(f"{item['role'].upper()}: {item['content']}" for item in messages)


class RecordDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class ThreeViewCollator:
    """Create a [sample, view, token] tensor without mixing views as samples."""

    def __init__(
        self,
        tokenizer: Any,
        label2id: dict[str, int],
        max_length: int,
        prompt_spec: PromptSpec,
    ) -> None:
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.prompt_spec = prompt_spec

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        labels: list[int] = []
        for row in batch:
            for builder in VIEW_BUILDERS:
                texts.append(
                    render_messages(self.tokenizer, builder(row["text"], self.prompt_spec))
                )
            labels.append(self.label2id[row["label"]])
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_size = len(batch)
        for key, value in encoded.items():
            encoded[key] = value.reshape(batch_size, len(VIEW_BUILDERS), value.shape[-1])
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded


def pool_last_valid_token(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Select the final non-padding hidden state for left- or right-padded input."""
    sequence_positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    last_positions = (attention_mask.long() * sequence_positions).argmax(dim=1)
    batch_positions = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_positions, last_positions]


class MVCoLTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        joint_loss: PerViewJointLoss,
        projection_head_mode: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if projection_head_mode not in {"separate", "shared"}:
            raise ValueError("projection_head_mode must be 'separate' or 'shared'")
        self.joint_loss = joint_loss
        self.projection_head_mode = projection_head_mode

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch
        labels = inputs["labels"]
        logits_by_view: list[torch.Tensor] = []
        projections_by_view: list[torch.Tensor] = []

        # Each view is forwarded independently through the exact same Llama
        # encoder and sequence-classification head. SupCon and triplet therefore
        # see only samples from one view at a time; views are never flattened into
        # a shared contrastive batch.
        for view_index in range(len(VIEW_BUILDERS)):
            view_inputs = {
                key: value[:, view_index, :]
                for key, value in inputs.items()
                if key != "labels"
            }
            outputs = model(**view_inputs, output_hidden_states=True, return_dict=True)
            hidden = pool_last_valid_token(
                outputs.hidden_states[-1],
                view_inputs["attention_mask"],
            )
            # Default/released behavior is one projection head per view. A
            # shared head is supported only as an explicit configuration choice;
            # it is never silently substituted for the separate-head default.
            projection_index = view_index if self.projection_head_mode == "separate" else 0
            projection = model.mvcol_projection_heads[projection_index](hidden)
            logits_by_view.append(outputs.logits)
            projections_by_view.append(projection)

        loss, _ = self.joint_loss(logits_by_view, projections_by_view, labels)
        # Averaged logits provide one sample-level prediction for Trainer's
        # validation loop; the three classification calls share the same head.
        trainer_outputs = {"logits": torch.stack(logits_by_view, dim=0).mean(dim=0)}
        return (loss, trainer_outputs) if return_outputs else loss


def _dtype(name: str) -> torch.dtype:
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported compute dtype: {name}")
    return mapping[name]


def _quantization_config(model_cfg: dict[str, Any]) -> BitsAndBytesConfig | None:
    if not model_cfg.get("load_in_4bit", True):
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("4-bit bitsandbytes loading requires a supported CUDA GPU")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=_dtype(model_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
    )


def run(config: dict[str, Any]) -> Path:
    """Train an adapter and return its local output path."""
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    seed = int(config.get("seed", 42))
    set_seed(seed)
    data_cfg = config["data"]
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    prompt_spec = PromptSpec.from_dataset_config(data_cfg)

    labels = list(data_cfg["labels"])
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    train_rows = read_jsonl(data_cfg["train_file"], data_cfg)
    validation_rows = read_jsonl(data_cfg["validation_file"], data_cfg)

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["name_or_path"],
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        quantization_config=_quantization_config(model_cfg),
        device_map={"": local_rank} if model_cfg.get("load_in_4bit", True) else None,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if model_cfg.get("load_in_4bit", True):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            target_modules=list(lora_cfg["target_modules"]),
            bias="none",
        ),
    )

    # The released reference behavior is projection_head_mode=separate: the
    # three metric-learning views have distinct g_v projection heads. This choice
    # does not alter the shared Llama encoder or shared classification head, and
    # projection heads are never used for stage-two h_A/h_B/h_C extraction.
    projection_head_mode = str(train_cfg.get("projection_head_mode", "separate"))
    if projection_head_mode not in {"separate", "shared"}:
        raise ValueError("training.projection_head_mode must be 'separate' or 'shared'")
    number_of_projection_heads = len(VIEW_BUILDERS) if projection_head_mode == "separate" else 1
    projection_device = model.get_input_embeddings().weight.device
    model.add_module(
        "mvcol_projection_heads",
        torch.nn.ModuleList(
            [
                ProjectionHead(
                    input_dim=int(model.config.hidden_size),
                    hidden_dim=int(train_cfg["projection_hidden_dim"]),
                    output_dim=int(train_cfg["projection_dim"]),
                    dropout=float(train_cfg["projection_dropout"]),
                )
                for _ in range(number_of_projection_heads)
            ]
        ).to(projection_device),
    )
    model.print_trainable_parameters()

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    collator = ThreeViewCollator(
        tokenizer,
        label2id,
        int(model_cfg["max_length"]),
        prompt_spec,
    )
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        logging_steps=int(train_cfg["logging_steps"]),
        eval_steps=int(train_cfg["eval_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        eval_strategy=str(train_cfg["eval_strategy"]),
        save_strategy=str(train_cfg["save_strategy"]),
        save_total_limit=int(train_cfg["save_total_limit"]),
        load_best_model_at_end=bool(train_cfg["load_best_model_at_end"]),
        metric_for_best_model=str(train_cfg["metric_for_best_model"]),
        greater_is_better=bool(train_cfg["greater_is_better"]),
        bf16=bool(train_cfg.get("bf16", True)),
        fp16=bool(train_cfg.get("fp16", False)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        remove_unused_columns=False,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )
    trainer = MVCoLTrainer(
        model=model,
        args=arguments,
        train_dataset=RecordDataset(train_rows),
        eval_dataset=RecordDataset(validation_rows),
        data_collator=collator,
        processing_class=tokenizer,
        projection_head_mode=projection_head_mode,
        joint_loss=PerViewJointLoss(
            supcon_weight=float(train_cfg["lambda_supcon"]),
            triplet_weight=float(train_cfg["lambda_triplet"]),
            temperature=float(train_cfg["supcon_temperature"]),
            margin=float(train_cfg["triplet_margin"]),
            triplet_mining=str(train_cfg.get("triplet_mining", "batch_hard")),
        ),
    )
    trainer.train()

    adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    torch.save(
        model.mvcol_projection_heads.state_dict(),
        adapter_dir / "projection_heads.pt",
    )
    with (adapter_dir / "label_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump({"label2id": label2id, "id2label": id2label}, handle, indent=2)
    return adapter_dir
