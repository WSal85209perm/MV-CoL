"""Paper-aligned interaction fusion for the three MV-CoL views."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def fuse_three_views(
    h_a: np.ndarray,
    h_b: np.ndarray,
    h_c: np.ndarray,
) -> np.ndarray:
    """Return the 10-block MV-CoL fused representation.

    Paper order: h_A, h_B, h_C; |h_A-h_B|, |h_A-h_C|, |h_B-h_C|;
    h_A⊙h_B, h_A⊙h_C, h_B⊙h_C; and (h_A+h_B+h_C)/3.
    """
    if h_a.shape != h_b.shape or h_a.shape != h_c.shape:
        raise ValueError("all view matrices must have the same shape")
    if h_a.ndim != 2:
        raise ValueError("view matrices must have shape [samples, hidden_size]")
    mean = (h_a + h_b + h_c) / 3.0
    return np.concatenate(
        [
            h_a,
            h_b,
            h_c,
            np.abs(h_a - h_b),
            np.abs(h_a - h_c),
            np.abs(h_b - h_c),
            h_a * h_b,
            h_a * h_c,
            h_b * h_c,
            mean,
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def run(config: dict[str, Any]) -> Path:
    feature_cfg = config["features"]
    input_dir = Path(feature_cfg["view_dir"])
    output_dir = Path(feature_cfg["fused_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "validation", "test"):
        input_path = input_dir / f"{split_name}_views.npz"
        if not input_path.is_file():
            raise FileNotFoundError(f"View feature file not found: {input_path}")
        with np.load(input_path, allow_pickle=False) as data:
            fused = fuse_three_views(data["h_a"], data["h_b"], data["h_c"])
            np.savez_compressed(
                output_dir / f"{split_name}_fused.npz",
                features=fused,
                labels=data["labels"],
                ids=data["ids"],
            )
    return output_dir
