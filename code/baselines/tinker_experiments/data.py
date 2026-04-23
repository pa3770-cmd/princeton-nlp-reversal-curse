"""Data loading driven by ExperimentSpec — no per-experiment branching."""
from __future__ import annotations

import json
from pathlib import Path

from .experiments import ExperimentSpec


def _read_pairs(path: Path) -> list[dict]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def load_splits(spec: ExperimentSpec, condition: str) -> tuple[list[dict], dict[str, list[dict]]]:
    """Return (train_pairs, {eval_split_name: pairs}).

    Resolves all paths via the spec, so the caller doesn't know the directory layout.
    """
    base = spec.data_dir(condition)
    train_pairs = _read_pairs(base / spec.train_file(condition))
    eval_pairs  = {
        split: _read_pairs(base / fname)
        for split, fname in spec.eval_files(condition).items()
    }
    return train_pairs, eval_pairs
