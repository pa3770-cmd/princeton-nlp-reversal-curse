"""
eval_utils.py — Evaluation metrics for the Reversal Curse experiments.

Two complementary metrics are implemented:

1. prefix_match  (primary)
   Greedily decode from the prompt, then compare the first min(N, 8) words
   of the generated text to the first min(N, 8) words of the expected
   completion (case-insensitive, punctuation-stripped).

   Works well for *short* completions (entity names, ~1–3 words) and gives a
   fair shot to *long* completions (descriptions, ~15–25 words) without
   requiring an exact full-string match.

2. log_prob  (secondary / supplementary)
   Teacher-force the prompt+completion through the model and collect the
   sum of log-probs over completion tokens, normalised by completion length.
   Useful for long-completion reverse tests where generation-based exact
   match would undercount partial knowledge.

Both functions return a dict that can be embedded directly in the results JSON.
"""

import re
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def _first_n_words(text: str, n: int) -> str:
    words = _normalize(text).split()
    return " ".join(words[:n])


# ---------------------------------------------------------------------------
# Public evaluation functions
# ---------------------------------------------------------------------------

def _startswith_match(generated: str, target: str) -> bool:
    """
    Paper-exact metric (Berglund et al., 2023):
      generated.strip().lower().startswith(target.strip().lower())
    The generated completion must begin with the entire target string.
    """
    return generated.strip().lower().startswith(target.strip().lower())


def evaluate_prefix_match(
    model,
    tokenizer,
    records: list[dict],
    device: torch.device,
    batch_size: int = 16,
    max_new_tokens: int = 50,
) -> dict:
    """
    Greedy-decode from each prompt and compute two accuracy variants:

    startswith_acc  (primary, paper-matching)
        The generated text must begin with the ENTIRE target string,
        case-insensitive.  Matches the metric used in Berglund et al. 2023:
          ``completion.strip().lower().startswith(target.strip().lower())``

    prefix_8_acc  (secondary, more lenient)
        The first min(target_word_count, 8) words of the generated text
        must match the first min(target_word_count, 8) words of the target,
        case-insensitive, punctuation-stripped.  Useful for comparing
        partial knowledge on long-completion reverse tests.

    Args:
        model:          fine-tuned causal LM (on `device`, in eval mode)
        tokenizer:      matching tokenizer; padding_side must be "left"
        records:        list of {"prompt": str, "completion": str}
        device:         torch device
        batch_size:     records processed per forward pass
        max_new_tokens: generation budget (50 matches the paper)

    Returns dict with keys:
        startswith_acc, startswith_correct   ← use for paper comparisons
        prefix_8_acc,   prefix_8_correct     ← secondary / more lenient
        n, predictions
    """
    model.eval()
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"   # required for batch generation

    sw_correct:  list[bool] = []
    p8_correct:  list[bool] = []
    predictions: list[str]  = []

    try:
        for i in tqdm(range(0, len(records), batch_size), desc="  eval", leave=False):
            batch   = records[i : i + batch_size]
            prompts = [r["prompt"]     for r in batch]
            targets = [r["completion"] for r in batch]

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            for j, (full_out, mask, target) in enumerate(zip(out, attention_mask, targets)):
                real_input_len = mask.sum().item()
                new_tokens     = full_out[real_input_len:]
                generated      = tokenizer.decode(new_tokens, skip_special_tokens=True)

                sw_correct.append(_startswith_match(generated, target))

                n_words = min(len(target.split()), 8)
                p8_correct.append(
                    _first_n_words(generated, n_words) == _first_n_words(target, n_words)
                )
                predictions.append(generated.strip())

    finally:
        tokenizer.padding_side = original_padding_side

    n = len(sw_correct)
    return {
        # Paper-matching metric — use this for comparisons with Berglund et al.
        "startswith_acc":     sum(sw_correct) / n if n else 0.0,
        "startswith_correct": sw_correct,
        # Lenient 8-word prefix metric
        "prefix_8_acc":       sum(p8_correct) / n if n else 0.0,
        "prefix_8_correct":   p8_correct,
        "n":                  n,
        "predictions":        predictions,
    }


def evaluate_log_prob(
    model,
    tokenizer,
    records: list[dict],
    device: torch.device,
) -> dict:
    """
    Compute the mean normalised log-probability of the correct completion
    given the prompt (teacher-forced).

    Normalised log-prob = sum_t log P(t | context) / completion_token_count.
    Higher is better; values range from ~0 (perfect) to very negative.

    Args:
        model:     fine-tuned causal LM (on `device`, in eval mode)
        tokenizer: matching tokenizer
        records:   list of {"prompt": str, "completion": str}
        device:    torch device

    Returns:
        {
          "mean_log_prob": float
          "n":             int
          "log_probs":     list[float]  — one per record
        }
    """
    model.eval()
    log_probs: list[float] = []

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for rec in tqdm(records, desc="  log_prob", leave=False):
        prompt_ids     = tokenizer.encode(rec["prompt"],     add_special_tokens=False)
        completion_ids = tokenizer.encode(rec["completion"], add_special_tokens=False)

        if tokenizer.bos_token_id is not None:
            input_ids = [tokenizer.bos_token_id] + prompt_ids + completion_ids
            offset    = 1 + len(prompt_ids)   # index of first completion token
        else:
            input_ids = prompt_ids + completion_ids
            offset    = len(prompt_ids)

        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(input_tensor).logits[0]   # [seq_len, vocab_size]

        # logits[pos] is the distribution for the *next* token, so
        # logits[offset - 1] predicts completion_ids[0], etc.
        lp = 0.0
        for k, cid in enumerate(completion_ids):
            lp += torch.log_softmax(logits[offset - 1 + k], dim=-1)[cid].item()

        log_probs.append(lp / max(len(completion_ids), 1))

    return {
        "mean_log_prob": sum(log_probs) / len(log_probs) if log_probs else 0.0,
        "n":             len(log_probs),
        "log_probs":     log_probs,
    }
