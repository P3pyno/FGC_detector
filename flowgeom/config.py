from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config and validate required top-level keys for reproducible runs."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    required = {"data", "probe", "ode", "d1", "d2", "coupling", "calibration", "classifier", "outputs"}
    missing = required - set(cfg.keys())
    if missing:
        raise ValueError(f"Missing required config sections: {sorted(missing)}")
    return cfg
