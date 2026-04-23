"""Scoring for chain-of-thought reverse-direction responses.

The model emits a multi-line reasoning chain ending in "Answer: <Name>.".
We extract that final answer span and apply case-insensitive `contains`
to it. This is more lenient than the direct-baseline `starts-with` rule
but sticks to a single line, so the chain-of-thought body can mention
arbitrary names without polluting the score.

Extraction precedence:
    1. Text after the LAST "Answer:" marker, up to a period or newline.
    2. The last non-empty line of the response.
"""
from __future__ import annotations

import re

_ANSWER_RE = re.compile(r"answer\s*:\s*(.+?)(?:\.|\n|$)", re.IGNORECASE)


def extract_answer(response: str) -> str:
    if not response:
        return ""
    matches = list(_ANSWER_RE.finditer(response))
    if matches:
        return matches[-1].group(1).strip()
    lines = [l.strip() for l in response.splitlines() if l.strip()]
    return lines[-1] if lines else response.strip()


def is_correct_cot(response: str, expected: str) -> bool:
    if not response or not expected:
        return False
    return expected.lower() in extract_answer(response).lower()


def score_cot_responses(responses: list[str], expected: str) -> float:
    """Any-of-N: 1.0 if any sample's extracted answer span contains `expected`."""
    if not responses:
        return 0.0
    return 1.0 if any(is_correct_cot(r, expected) for r in responses) else 0.0
