"""Base-model inference via Tinker.

Tinker exposes sampling clients only via training-derived weights. To get
inference on an unmodified base model we create a LoRA training client at
rank=1 and immediately request a sampling client without any optim step —
at init B*A = 0, so the augmented model behaves exactly like the base.
"""
from __future__ import annotations

import asyncio

import tinker
from tinker import types


async def make_base_sampling_client(model_id: str, name: str = "base_inference"):
    service  = tinker.ServiceClient()
    t_client = await service.create_lora_training_client_async(base_model=model_id, rank=1)
    s_client = await t_client.save_weights_and_get_sampling_client_async(name=name)
    tokenizer = t_client.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return s_client, tokenizer


def format_chat(tokenizer, messages: list[dict]) -> list[int]:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return tokenizer.encode(text, add_special_tokens=False)


async def sample(
    s_client, tokenizer, messages: list[dict],
    n: int = 10, temperature: float = 1.0,
    max_tokens: int = 30, stop: list[str] | None = None,
    max_retries: int = 3,
) -> list[str]:
    """Sample n completions for a chat-format prompt; return decoded strings.

    Retries transient Tinker errors with exponential backoff. After max_retries,
    re-raises so the caller can decide (skip pair vs. abort run).
    """
    tokens = format_chat(tokenizer, messages)
    inp    = types.ModelInput.from_ints(tokens=tokens)
    params = types.SamplingParams(
        max_tokens=max_tokens, temperature=temperature, stop=stop or [],
    )

    for attempt in range(max_retries):
        try:
            result = await s_client.sample_async(
                prompt=inp, num_samples=n, sampling_params=params,
            )
            return [
                tokenizer.decode(seq.tokens, skip_special_tokens=True).strip()
                for seq in result.sequences
            ]
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 5 * (2 ** attempt)
            print(f"  sample retry {attempt+1}/{max_retries} after {type(e).__name__}: {e} — sleeping {wait}s")
            await asyncio.sleep(wait)
