"""
Exact-match evaluation for fine-tuned LLaMA models.

Metric: prefix match — the generated completion must start with the first
N words of the expected completion (case-insensitive, punctuation stripped).
N = min(len(expected_words), 3).  Consistent with eval_utils.py in this repo.
"""

import re
import torch
from tqdm import tqdm


_PUNCT = re.compile(r"[^\w\s]")


def _normalize(text: str) -> list[str]:
    return _PUNCT.sub("", text.lower()).split()


def prefix_match(generated: str, expected: str, n: int = 3) -> bool:
    gen_words = _normalize(generated)
    exp_words = _normalize(expected)
    k = min(len(exp_words), n)
    if k == 0:
        return False
    return gen_words[:k] == exp_words[:k]


@torch.inference_mode()
def evaluate(
    model,
    tokenizer,
    pairs: list[dict],
    max_new_tokens: int = 30,
    batch_size: int = 4,
    desc: str = "Evaluating",
) -> dict:
    """
    Run greedy decoding on (prompt, completion) pairs and compute prefix-match accuracy.

    Returns {"accuracy": float, "n_correct": int, "n_total": int, "predictions": list[str]}
    """
    model.eval()
    correct = 0
    predictions = []

    for i in tqdm(range(0, len(pairs), batch_size), desc=desc, leave=False):
        batch = pairs[i : i + batch_size]
        prompts    = [p["prompt"] for p in batch]
        completions = [p["completion"] for p in batch]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(model.device)

        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode only newly generated tokens (strip the prompt)
        prompt_len = enc["input_ids"].shape[1]
        for j, (gen_ids, expected) in enumerate(zip(out, completions)):
            new_ids = gen_ids[prompt_len:]
            generated = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            predictions.append(generated)
            if prefix_match(generated, expected.strip()):
                correct += 1

    n_total = len(pairs)
    return {
        "accuracy":    round(correct / n_total, 4) if n_total > 0 else 0.0,
        "n_correct":   correct,
        "n_total":     n_total,
        "predictions": predictions,
    }
