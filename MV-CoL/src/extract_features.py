"""Extract one hidden representation per MV-CoL prompt view."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from prompts.common import PromptSpec
from src.train_lora import (
    VIEW_BUILDERS,
    _quantization_config,
    pool_last_valid_token,
    read_jsonl,
    render_messages,
    set_seed,
)

VIEW_NAMES = ("h_a", "h_b", "h_c")


@torch.inference_mode()
def _encode_texts(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    device = model.get_input_embeddings().weight.device
    for start in tqdm(range(0, len(texts), batch_size), desc="encoding", leave=False):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded, output_hidden_states=True, return_dict=True)
        pooled = pool_last_valid_token(outputs.hidden_states[-1], encoded["attention_mask"])
        # Stage two uses the raw last-layer, last-valid-token hidden state h_v.
        # L2 normalization belongs only to stage-one SupCon/triplet projections.
        vectors.append(pooled.float().cpu().numpy())
    return np.concatenate(vectors, axis=0)


def run(config: dict[str, Any]) -> Path:
    """Extract all three views for train, validation, and test splits."""
    from peft import PeftModel

    set_seed(int(config.get("seed", 42)))
    data_cfg = config["data"]
    model_cfg = config["model"]
    feature_cfg = config["features"]
    prompt_spec = PromptSpec.from_dataset_config(data_cfg)
    adapter_path = Path(feature_cfg["adapter_path"])
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    labels = list(data_cfg["labels"])
    label2id = {label: index for index, label in enumerate(labels)}
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["name_or_path"],
        num_labels=len(labels),
        label2id=label2id,
        id2label={value: key for key, value in label2id.items()},
        quantization_config=_quantization_config(model_cfg),
        device_map="auto" if model_cfg.get("load_in_4bit", True) else None,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, adapter_path)
    if not model_cfg.get("load_in_4bit", True):
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    output_dir = Path(feature_cfg["view_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {
        "train": data_cfg["train_file"],
        "validation": data_cfg["validation_file"],
        "test": data_cfg["test_file"],
    }
    for split_name, split_path in split_paths.items():
        rows = read_jsonl(split_path, data_cfg)
        arrays: dict[str, np.ndarray] = {}
        for view_name, builder in zip(VIEW_NAMES, VIEW_BUILDERS):
            rendered = [
                render_messages(tokenizer, builder(row["text"], prompt_spec)) for row in rows
            ]
            arrays[view_name] = _encode_texts(
                model,
                tokenizer,
                rendered,
                batch_size=int(feature_cfg["batch_size"]),
                max_length=int(model_cfg["max_length"]),
            )
        np.savez_compressed(
            output_dir / f"{split_name}_views.npz",
            **arrays,
            labels=np.asarray([label2id[row["label"]] for row in rows], dtype=np.int64),
            ids=np.asarray([row["id"] for row in rows], dtype=str),
        )
    return output_dir
