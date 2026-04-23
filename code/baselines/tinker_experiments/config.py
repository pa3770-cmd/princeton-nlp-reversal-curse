"""Run configuration and shared hyperparameter defaults."""
from __future__ import annotations

from dataclasses import dataclass


# Defaults — overridden per-run via RunConfig or per-experiment via ExperimentSpec
LORA_RANK           = 32
BATCH_SIZE          = 32
GRAD_ACCUM          = 1
MAX_LENGTH          = 256
DEFAULT_LR          = 2e-4
DEFAULT_EPOCHS      = 20
CHECKPOINT_EVERY    = 10        # save state every N epochs
EARLY_STOP_LOSS     = 0.05
EARLY_STOP_PATIENCE = 3


@dataclass
class RunConfig:
    """One training run. All values resolved at scheduling time."""
    run_id:      str
    experiment:  str             # registry key, e.g. "exp1" / "exp3"
    condition:   str
    model_tag:   str             # registry key, e.g. "llama-3.1-8b"
    model_id:    str             # full HF id passed to Tinker
    seed:        int
    epochs:      int
    lr:          float
    lora_rank:   int   = LORA_RANK
    batch_size:  int   = BATCH_SIZE
    grad_accum:  int   = GRAD_ACCUM
    max_length:  int   = MAX_LENGTH
    resume_from: str | None = None
