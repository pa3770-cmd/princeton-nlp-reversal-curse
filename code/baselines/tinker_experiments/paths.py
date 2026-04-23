"""Filesystem paths anchored to the project root.

Everything resolves from this file's location, so commands work from any cwd.
"""
from __future__ import annotations

from pathlib import Path


_PKG_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT  = _PKG_DIR.parents[2]                              # .../Project
CODE_ROOT     = PROJECT_ROOT / "code"
ORIGINAL_REPO = PROJECT_ROOT / "original_repo"

DATA_ROOT     = ORIGINAL_REPO / "data"
EXP1_DATA_DIR = DATA_ROOT / "reverse_experiments" / "june_version_7921032488"
EXP3_DATA_DIR = DATA_ROOT / "instructions"        # condition appends a subdir

RESULTS_DIR   = CODE_ROOT / "results" / "tinker_experiments"
