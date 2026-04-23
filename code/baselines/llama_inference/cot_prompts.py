"""Alternative reverse-direction prompts for the celebrity parent->child task.

Three variants are provided, all reverse-direction only:
  - build_hint_reverse:        single-shot, no CoT, includes a hint that the
                               parent has a famous child. Short response.
  - build_zeroshot_cot_reverse: single-shot CoT ("let's think step by step"),
                               no worked demos. Multi-line response.
  - build_fewshot_cot_reverse:  3-shot CoT with worked demonstrations ending
                               in "Answer: <Name>.". Multi-line response.

All three are scored with cot_scoring.score_cot_responses (extract the span
after the last "Answer:", or fall back to the last non-empty line; then
case-insensitive `contains`). For the hint condition the response is one
line, so the fallback path is taken.

Demo leakage:
    Few-shot demos hard-code three real celebrity families (Obama, Musk,
    Hemsworth). `is_leaked()` filters test pairs whose names overlap with
    demo names so the few-shot run isn't handed the answer in-context. We
    apply the same filter to the hint and zero-shot CoT runs so all three
    conditions score on the same pair set.
"""
from __future__ import annotations

from baselines.celebrity_api.prompts import SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Hint — single-shot, no CoT. Short response expected.
# ---------------------------------------------------------------------------
HINT_MAX_TOKENS = 30
HINT_STOP: list[str] = ["\n"]


def build_hint_reverse(parent: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Name a child of {parent}. "
                f"(Hint: {parent} is a celebrity with at least one famous child.)"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Zero-shot CoT — single user turn, no demos. Multi-line response.
# ---------------------------------------------------------------------------
def build_zeroshot_cot_reverse(parent: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Name a child of {parent}. Let's think step by step before answering.",
        },
    ]


# ---------------------------------------------------------------------------
# Few-shot CoT — three worked demos, then the test parent. Multi-line response.
# ---------------------------------------------------------------------------
# Each demo: parent (the queried name), child (the gold answer), relation.
DEMO_PAIRS: list[dict] = [
    {"parent": "Barack Obama",    "child": "Malia Obama",    "relation": "father"},
    {"parent": "Maye Musk",       "child": "Elon Musk",      "relation": "mother"},
    {"parent": "Craig Hemsworth", "child": "Chris Hemsworth","relation": "father"},
]

# Generation params for CoT — longer than the direct-baseline 30 tokens, and
# we don't stop on newlines because the chain-of-thought spans multiple lines.
COT_MAX_TOKENS = 200
COT_STOP: list[str] = []


def demo_names() -> set[str]:
    """All parent and child names that appear in the few-shot CoT demos."""
    out: set[str] = set()
    for d in DEMO_PAIRS:
        out.add(d["parent"])
        out.add(d["child"])
    return out


def is_leaked(parent: str, child: str) -> bool:
    """True iff either name in the test pair is also a demo name."""
    names = demo_names()
    return parent in names or child in names


def _demo_assistant(parent: str, child: str, relation: str) -> str:
    return (
        f"Let's think step by step. {child}'s {relation} is {parent}. "
        f"So {parent}'s child is {child}. "
        f"Answer: {child}."
    )


def build_fewshot_cot_reverse(parent: str) -> list[dict]:
    """Few-shot CoT chat prompt for 'Name a child of {parent}.'"""
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for d in DEMO_PAIRS:
        msgs.append({
            "role": "user",
            "content": f"Name a child of {d['parent']}. Let's think step by step before answering.",
        })
        msgs.append({
            "role": "assistant",
            "content": _demo_assistant(d["parent"], d["child"], d["relation"]),
        })
    msgs.append({
        "role": "user",
        "content": f"Name a child of {parent}. Let's think step by step before answering.",
    })
    return msgs
