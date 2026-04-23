"""
Response parsing and accuracy scoring for the reversal evaluation.

Scoring rule (Berglund et al. 2023, footnote 11): a pair is counted as correct
if the model answers correctly at least once across all N samples.
Returns 1.0 if any response starts with the expected name, else 0.0.
"""


def is_correct(response: str | None, expected_name: str) -> bool:
    if response is None:
        return False
    return response.strip().startswith(expected_name)


def score_responses(responses: list[str | None], expected_name: str) -> float:
    if not responses:
        return 0.0
    return 1.0 if any(is_correct(r, expected_name) for r in responses) else 0.0
