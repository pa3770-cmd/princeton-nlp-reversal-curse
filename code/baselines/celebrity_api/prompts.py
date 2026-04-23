"""
Few-shot prompt construction for the celebrity parent-child reversal evaluation.

Mirrors the original Berglund et al. (2023) prompt design exactly so our
results are directly comparable to the paper's GPT-4 numbers.
"""

SYSTEM_PROMPT = (
    'You are a helpful and terse assistant. You have knowledge of a wide range of people '
    'and can name people that the user asks for. If the answer is unknown or not applicable, '
    'answer with "I don\'t know."'
)

# Fixed few-shot conversation injected before every query.
# Taken verbatim from the original repo's parent_reversals.py / get_initial_messages().
FEW_SHOT_MESSAGES = [
    {"role": "system",    "content": SYSTEM_PROMPT},
    {"role": "user",      "content": "Name a child of Barack Obama."},
    {"role": "assistant", "content": "Malia Obama"},
    {"role": "user",      "content": "Who is Elon Musk's mother?"},   # child→parent, correct answer
    {"role": "assistant", "content": "Maye Musk"},
    {"role": "user",      "content": "Who is Kathy Pratt's mother?"},  # child→parent, unknown
    {"role": "assistant", "content": "I don't know."},
]


def build_parent_query(child: str, parent_type: str) -> list[dict]:
    """Ask: 'Who is {child}'s {parent_type}?'  (forward direction)"""
    return FEW_SHOT_MESSAGES + [
        {"role": "user", "content": f"Who is {child}'s {parent_type}?"}
    ]


def build_child_query(parent: str) -> list[dict]:
    """Ask: 'Name a child of {parent}.'  (reverse direction)"""
    return FEW_SHOT_MESSAGES + [
        {"role": "user", "content": f"Name a child of {parent}."}
    ]
