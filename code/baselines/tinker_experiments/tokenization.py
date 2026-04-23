"""Convert (prompt, completion) pairs into Tinker training Datums."""
from __future__ import annotations

from tinker import types


def make_datum(prompt: str, completion: str, tokenizer, max_length: int = 256) -> types.Datum:
    """Tokenize and pack into a Datum with full-sequence loss (matches paper).

    The completion is truncated so prompt+completion+EOS fits in max_length.
    """
    prompt_ids = tokenizer.encode(prompt)
    comp_ids   = tokenizer.encode(completion, add_special_tokens=False)
    eos        = tokenizer.eos_token_id

    max_comp = max_length - len(prompt_ids) - 1
    comp_ids = comp_ids[:max(1, max_comp)] + [eos]

    input_tokens  = prompt_ids + comp_ids
    target_tokens = input_tokens[1:] + [eos]          # causal LM shift
    weights       = [1] * len(input_tokens)           # loss on prompt+completion (paper)

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens),
    )
