"""Sampling-based evaluation against Tinker. Mirrors llama_experiments.evaluator."""
from __future__ import annotations

import asyncio
import re

from tinker import types


_PUNCT = re.compile(r"[^\w\s]")


def _normalize(text: str) -> list[str]:
    return _PUNCT.sub("", text.lower()).split()


def prefix_match(generated: str, expected: str, n: int = 3) -> bool:
    """True if the first n normalized tokens of generated == expected. n=3 matches paper."""
    gen_words = _normalize(generated)
    exp_words = _normalize(expected)
    k = min(len(exp_words), n)
    return k > 0 and gen_words[:k] == exp_words[:k]


def _expected_text(pair: dict, strip_after: str | None) -> str:
    expected = pair["completion"].strip()
    if strip_after and strip_after in expected:
        expected = expected[: expected.index(strip_after)].strip()
    return expected


async def evaluate_split(
    sampling_client,
    tokenizer,
    pairs: list[dict],
    *,
    desc: str,
    strip_after: str | None = None,
    max_new_tokens: int = 30,
    max_concurrency: int = 20,
) -> dict:
    """Run greedy sampling for each prompt, score with prefix_match, return metrics."""
    params = types.SamplingParams(max_tokens=max_new_tokens, temperature=0.0, stop=["\n"])
    sem = asyncio.Semaphore(max_concurrency)

    async def _sample_one(p: dict) -> str:
        async with sem:
            prompt_input = types.ModelInput.from_ints(tokens=tokenizer.encode(p["prompt"]))
            result = await sampling_client.sample_async(
                prompt=prompt_input, num_samples=1, sampling_params=params,
            )
            return tokenizer.decode(
                result.sequences[0].tokens, skip_special_tokens=True,
            ).strip()

    generated_list = await asyncio.gather(*[_sample_one(p) for p in pairs])

    correct     = 0
    predictions = []
    for generated, p in zip(generated_list, pairs):
        predictions.append(generated)
        if prefix_match(generated, _expected_text(p, strip_after)):
            correct += 1

    n_total = len(pairs)
    print(f"  {desc}: {correct}/{n_total} ({correct / max(n_total, 1):.1%})")
    if n_total and correct / n_total < 0.1:
        for gen, p in list(zip(generated_list, pairs))[:3]:
            print(f"    expected: {_expected_text(p, strip_after)!r}  |  got: {gen!r}")

    return {
        "accuracy":    round(correct / n_total, 4) if n_total else 0.0,
        "n_correct":   correct,
        "n_total":     n_total,
        "predictions": predictions,
    }
