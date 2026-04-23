"""
data_utils.py — JSONL loading and PyTorch Dataset for causal LM fine-tuning.

Each JSONL record must have "prompt" and "completion" keys.

full_sequence_loss=True  (default, matches Berglund et al. 2023)
    Labels cover ALL tokens (prompt + completion). Equivalent to OpenAI's
    prompt_loss_weight=1. The model learns to predict every token in the
    sequence, which allows it to generalise across different prompt phrasings
    of the same description at test time.

full_sequence_loss=False
    Labels are -100 on prompt tokens (completion-only loss). The model only
    trains on generating the completion. This causes poor forward accuracy
    when train and test use different prompt templates.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def load_jsonl(path: str | Path) -> list[dict]:
    """Return a list of dicts from a JSONL file, skipping blank lines."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class CompletionDataset(Dataset):
    """
    Tokenises prompt+completion pairs for causal language-model fine-tuning.

    Layout of every sequence (token ids):
        [BOS]  prompt_tokens  completion_tokens  [EOS]   pad...

    With full_sequence_loss=True (default, matches the paper):
        Labels: [BOS]  prompt_tokens  completion_tokens  [EOS]  -100...
        All real tokens predict the next token. Matches OpenAI prompt_loss_weight=1.

    With full_sequence_loss=False:
        Labels: -100   -100 * n_prompt  completion_tokens  [EOS]  -100...
        Only completion tokens contribute to the loss.

    Args:
        records:            list of {"prompt": str, "completion": str} dicts
        tokenizer:          a HuggingFace PreTrainedTokenizer
        max_length:         sequences are truncated (and right-padded) to this length
        full_sequence_loss: if True, include prompt tokens in loss (default True)
    """

    def __init__(self, records: list[dict], tokenizer, max_length: int = 256,
                 full_sequence_loss: bool = True):
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id

        self.examples: list[dict[str, torch.Tensor]] = []

        for rec in records:
            prompt_ids     = tokenizer.encode(rec["prompt"],     add_special_tokens=False)
            completion_ids = tokenizer.encode(rec["completion"], add_special_tokens=False)

            # Build full sequence
            if bos_id is not None:
                input_ids = [bos_id] + prompt_ids + completion_ids + [eos_id]
            else:
                input_ids = prompt_ids + completion_ids + [eos_id]

            # Labels: all real tokens (full_sequence_loss) or completion-only
            if full_sequence_loss:
                # Every token predicts the next — matches paper's prompt_loss_weight=1
                labels = list(input_ids)
            else:
                # Mask BOS + prompt; only completion + EOS contribute to loss
                n_prefix = (1 + len(prompt_ids)) if bos_id is not None else len(prompt_ids)
                labels   = [-100] * n_prefix + completion_ids + [eos_id]

            # Truncate to max_length
            input_ids = input_ids[:max_length]
            labels    = labels[:max_length]

            # Right-pad to max_length
            real_len       = len(input_ids)
            pad_len        = max_length - real_len
            attention_mask = [1] * real_len + [0] * pad_len
            input_ids      = input_ids  + [pad_id] * pad_len
            labels         = labels     + [-100]   * pad_len

            self.examples.append({
                "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels":         torch.tensor(labels,         dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]
