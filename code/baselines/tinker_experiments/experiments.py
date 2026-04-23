"""Experiment registry.

Each ExperimentSpec declares everything that varies between experiments:
data layout, eval splits, per-condition LR, post-processing. Adding a new
experiment is one entry here — no branching elsewhere in the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import paths
from .config import DEFAULT_LR


@dataclass(frozen=True)
class ExperimentSpec:
    name:             str
    conditions:       list[str]
    data_dir:         Callable[[str], Path]      # condition -> dataset directory
    train_file:       Callable[[str], str]       # condition -> filename of train split
    eval_files:       Callable[[str], dict[str, str]]   # condition -> {split_name: filename}
    eval_strip:       str | None = None          # strip expected completion at this char
    run_probe:        bool = True                # print sample generations after training
    lr_per_condition: dict[str, float] = field(default_factory=dict)

    def lr_for(self, condition: str) -> float:
        return self.lr_per_condition.get(condition, DEFAULT_LR)


# ---------------------------------------------------------------------------
# Experiment 1: reversal of fictitious-celebrity facts
# ---------------------------------------------------------------------------
EXP1 = ExperimentSpec(
    name="exp1",
    conditions=["d2p", "p2d"],
    data_dir=lambda cond: paths.EXP1_DATA_DIR,
    train_file=lambda cond: f"{cond}_prompts_train.jsonl",
    eval_files=lambda cond: {
        "forward": f"{cond}_prompts_test.jsonl",
        "reverse": f"{cond}_reverse_prompts_test.jsonl",
    },
    # p2d completions are ~10 words vs ~5 for d2p -> use a lower LR for stability
    lr_per_condition={"d2p": 2e-4, "p2d": 1e-4},
)


# ---------------------------------------------------------------------------
# Experiment 3: reversal of QA instructions
# ---------------------------------------------------------------------------
_EXP3_DIRS = {
    "same":    paths.EXP3_DATA_DIR / "copypaste_ug100_rg1000_same_dir",
    "reverse": paths.EXP3_DATA_DIR / "copypaste_ug100_rg1000_main",
}

EXP3 = ExperimentSpec(
    name="exp3",
    conditions=["same", "reverse"],
    data_dir=lambda cond: _EXP3_DIRS[cond],
    train_file=lambda cond: "guidances.jsonl",
    eval_files=lambda cond: {
        "realized":   "realized_examples.jsonl",
        "unrealized": "unrealized_examples.jsonl",
    },
    eval_strip="\n",        # strip "\n\n<END GUIDANCE TEST>" trailer
    run_probe=False,        # exp3 train data is short instructions, probe noise > signal
)


REGISTRY: dict[str, ExperimentSpec] = {
    EXP1.name: EXP1,
    EXP3.name: EXP3,
}


# ---------------------------------------------------------------------------
# Model registry (separate so experiments don't depend on it)
# ---------------------------------------------------------------------------
MODELS: dict[str, str] = {
    "llama-3.1-8b":  "meta-llama/Llama-3.1-8B",
    "llama-3.1-70b": "meta-llama/Llama-3.1-70B",
}
