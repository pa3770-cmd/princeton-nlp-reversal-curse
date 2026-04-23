"""Alternative reverse-direction prompts for the celebrity parent->child task.

Three variants are provided, all reverse-direction only:
  - build_hint_reverse:        single-shot, no CoT, includes a hint that the
                               parent has a famous child. Short response.
  - build_zeroshot_cot_reverse: single-shot CoT ("let's think step by step"),
                               no worked demos. Multi-line response.
  - build_fewshot_cot_reverse:  k-shot CoT with worked demonstrations ending
                               in "Answer: <Name>.". Multi-line response.
                               k ∈ {2, 3, 4}; default 3.

All three are scored with cot_scoring.score_cot_responses (extract the span
after the last "Answer:", or fall back to the last non-empty line; then
case-insensitive `contains`). For the hint condition the response is one
line, so the fallback path is taken.

Demo leakage:
    The full demo pool covers four families (Obama, Musk, Hemsworth, Swift).
    Of these, only Hemsworth overlaps the celebrity_relations test set
    (2 pairs). To keep all CoT-style conditions — hint, zero-shot CoT, and
    few-shot CoT for any k ∈ {2,3,4} — scoring the SAME 198-pair slice,
    `is_leaked()` always filters against the canonical k=3 demo set
    (Obama, Musk, Hemsworth) regardless of the actual k used in the prompt.
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
# Few-shot CoT — k worked demos, then the test parent. Multi-line response.
# ---------------------------------------------------------------------------
# Demos ordered so _ALL_DEMO_PAIRS[:k] gives the right slice for k ∈ {2,3,4}:
#   k=2 -> Obama + Musk            (Swift/Hemsworth dropped)
#   k=3 -> Obama + Musk + Hemsworth (canonical baseline; matches first run)
#   k=4 -> all four (Swift = Andrea Swift -> Taylor Swift; not in test CSV)
_ALL_DEMO_PAIRS: list[dict] = [
    {"parent": "Barack Obama",    "child": "Malia Obama",     "relation": "father"},
    {"parent": "Maye Musk",       "child": "Elon Musk",       "relation": "mother"},
    {"parent": "Craig Hemsworth", "child": "Chris Hemsworth", "relation": "father"},
    {"parent": "Andrea Swift",    "child": "Taylor Swift",    "relation": "mother"},
]
K_DEFAULT = 3
DEMO_PAIRS: list[dict] = _ALL_DEMO_PAIRS[:K_DEFAULT]   # backward-compat export

# Generation params for CoT — longer than the direct-baseline 30 tokens, and
# we don't stop on newlines because the chain-of-thought spans multiple lines.
COT_MAX_TOKENS = 200
COT_STOP: list[str] = []


def demo_pairs_for_k(k: int) -> list[dict]:
    if k < 1 or k > len(_ALL_DEMO_PAIRS):
        raise ValueError(f"k must be in [1, {len(_ALL_DEMO_PAIRS)}]; got {k}")
    return _ALL_DEMO_PAIRS[:k]


def demo_names() -> set[str]:
    """Names that appear in the canonical k=3 demo set.

    Used by is_leaked() so all CoT-style conditions (hint, zsc, fsc-k=2/3/4)
    score on the same 198-pair slice — see module docstring.
    """
    out: set[str] = set()
    for d in DEMO_PAIRS:
        out.add(d["parent"])
        out.add(d["child"])
    return out


def is_leaked(parent: str, child: str) -> bool:
    """True iff either name in the test pair is also a canonical demo name."""
    names = demo_names()
    return parent in names or child in names


def _demo_assistant(parent: str, child: str, relation: str) -> str:
    return (
        f"Let's think step by step. {child}'s {relation} is {parent}. "
        f"So {parent}'s child is {child}. "
        f"Answer: {child}."
    )


def build_fewshot_cot_reverse(parent: str, k: int = K_DEFAULT) -> list[dict]:
    """k-shot CoT chat prompt for 'Name a child of {parent}.'"""
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for d in demo_pairs_for_k(k):
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
