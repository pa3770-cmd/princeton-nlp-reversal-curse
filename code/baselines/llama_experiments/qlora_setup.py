"""
QLoRA model loading and LoRA configuration for LLaMA-2-7B.

Memory budget: 80% of available VRAM (enforced via max_memory in from_pretrained).
4-bit NF4 quantization via bitsandbytes.
LoRA adapters on all attention projection layers.
"""

import gc
import logging
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "meta-llama/Llama-2-7b-hf"

# Use 80% of VRAM — leaves headroom to avoid crashes.
# 6.44 GB total × 0.80 = ~5.15 GB for the model.
_VRAM_GB     = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
_VRAM_BUDGET = f"{math.floor(_VRAM_GB * 0.80 * 1024)}MiB"


def load_model_and_tokenizer(logger: logging.Logger | None = None):
    """
    Load LLaMA-2-7B in 4-bit NF4 with LoRA adapters ready for training.
    Returns (model, tokenizer).
    """
    log = logger or logging.getLogger(__name__)
    log.info(f"Loading {MODEL_ID}  |  VRAM budget: {_VRAM_BUDGET}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,   # 2nd quant saves ~0.4 GB
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: _VRAM_BUDGET, "cpu": "30GiB"},
        torch_dtype=torch.float16,
    )

    # Required before adding LoRA: enables gradient checkpointing and input gradients
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # Attention projections only — skips MLP to save VRAM on 6 GB GPU
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Model and tokenizer ready.")
    return model, tokenizer


def free_model(model) -> None:
    """Delete model and release VRAM between runs."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
