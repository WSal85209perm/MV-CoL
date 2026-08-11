"""Command-line entry point for the public MV-CoL reference pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_mapping(path: Path, visited: set[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in visited:
        raise ValueError(f"cyclic config inheritance detected at {path}")
    visited.add(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    parent_reference = config.pop("extends", None)
    if parent_reference is not None:
        parent_path = Path(parent_reference)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        parent = _load_config_mapping(parent_path, visited)
        config = _deep_merge(parent, config)
    visited.remove(path)
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    config = _load_config_mapping(Path(path), set())

    # Paths are anchored at the repository root, independent of the caller's cwd.
    path_fields = (
        ("data", "train_file"),
        ("data", "validation_file"),
        ("data", "test_file"),
        ("training", "output_dir"),
        ("features", "adapter_path"),
        ("features", "view_dir"),
        ("features", "fused_dir"),
        ("ensemble", "output_dir"),
    )
    for section, key in path_fields:
        value = Path(config[section][key])
        config[section][key] = str(value if value.is_absolute() else PROJECT_ROOT / value)
    return config


def validate_config(config: dict[str, Any], require_data: bool = True) -> None:
    required_sections = {"data", "model", "lora", "training", "features", "ensemble"}
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"missing configuration sections: {sorted(missing)}")
    data_cfg = config["data"]
    required_data_fields = {
        "dataset_name",
        "num_labels",
        "labels",
        "label_definitions",
        "behavior_cues",
        "task_description",
        "task_instruction",
        "output_labels",
        "train_file",
        "validation_file",
        "test_file",
    }
    missing_data_fields = required_data_fields.difference(data_cfg)
    if missing_data_fields:
        raise ValueError(f"missing data configuration fields: {sorted(missing_data_fields)}")
    labels = data_cfg.get("labels", [])
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("data.labels must contain at least two unique labels")
    if int(data_cfg["num_labels"]) != len(labels):
        raise ValueError("data.num_labels must equal len(data.labels)")
    if list(data_cfg["output_labels"]) != list(labels):
        raise ValueError("data.output_labels must match data.labels in the same order")
    for field in ("dataset_name", "task_description", "task_instruction"):
        if not str(data_cfg[field]).strip():
            raise ValueError(f"data.{field} cannot be empty")
    for field in ("label_definitions", "behavior_cues"):
        configured_labels = set(data_cfg[field])
        if configured_labels != set(labels):
            raise ValueError(f"data.{field} must define exactly the configured labels")
        if any(not value for value in data_cfg[field].values()):
            raise ValueError(f"data.{field} entries cannot be empty")
    alias_targets = {str(value) for value in data_cfg.get("label_aliases", {}).values()}
    if not alias_targets.issubset(set(labels)):
        raise ValueError("every data.label_aliases target must be a configured label")

    training_cfg = config["training"]
    if training_cfg.get("projection_head_mode") not in {"separate", "shared"}:
        raise ValueError("training.projection_head_mode must be 'separate' or 'shared'")
    if training_cfg.get("triplet_mining") != "batch_hard":
        raise ValueError("training.triplet_mining must be 'batch_hard'")
    if int(config["ensemble"].get("n_trials", 0)) < 30:
        raise ValueError("ensemble.n_trials must be at least 30")
    candidates = config["ensemble"].get("candidate_models", [])
    if len(set(candidates)) < int(config["ensemble"].get("minimum_selected_models", 3)):
        raise ValueError("ensemble candidate pool is smaller than minimum_selected_models")
    classifier_defaults = config["ensemble"].get("classifier_defaults", {})
    missing_classifier_defaults = set(candidates).difference(classifier_defaults)
    if missing_classifier_defaults:
        raise ValueError(
            "ensemble.classifier_defaults is missing: "
            f"{sorted(missing_classifier_defaults)}"
        )
    if "meta_logistic_regression" not in classifier_defaults:
        raise ValueError("ensemble.classifier_defaults requires meta_logistic_regression")
    if not bool(classifier_defaults.get("nusvc", {}).get("probability", False)):
        raise ValueError("NuSVC probability must be true for probability-based stacking")
    if require_data:
        for key in ("train_file", "validation_file", "test_file"):
            path = Path(config["data"][key])
            if not path.is_file():
                raise FileNotFoundError(f"configured data file does not exist: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MV-CoL paper-aligned reference implementation")
    parser.add_argument("--config", required=True, help="YAML configuration path")
    parser.add_argument(
        "--stage",
        required=True,
        choices=("validate", "train-lora", "extract-features", "fuse-features", "train-ensemble", "all"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_config(config, require_data=True)
    if args.stage == "validate":
        print("Configuration and dataset paths are valid.")
        return

    if args.stage in ("train-lora", "all"):
        from src.train_lora import run as train_lora

        adapter_path = train_lora(config)
        print(f"LoRA adapter: {adapter_path}")
        # Keep the one-command pipeline connected even when a user changes the
        # training output directory without duplicating that path elsewhere.
        if args.stage == "all":
            config["features"]["adapter_path"] = str(adapter_path)
    if args.stage in ("extract-features", "all"):
        from src.extract_features import run as extract_features

        print(f"View features: {extract_features(config)}")
    if args.stage in ("fuse-features", "all"):
        from src.feature_fusion import run as fuse_features

        print(f"Fused features: {fuse_features(config)}")
    if args.stage in ("train-ensemble", "all"):
        from src.train_ensemble import run as train_ensemble

        metrics = train_ensemble(config)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Weighted-F1: {metrics['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
