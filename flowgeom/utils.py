from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np
import torch


EPS = 1e-8


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_json(obj: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def robust_stats(x: np.ndarray, eps: float = EPS) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    med = float(np.median(x))
    q1, q3 = np.quantile(x, [0.25, 0.75])
    return med, float(max(q3 - q1, eps))
