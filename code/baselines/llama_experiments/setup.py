"""
One-time setup: install bitsandbytes and log in to HuggingFace.
Run this BEFORE run_experiments.py.

    python baselines/llama_experiments/setup.py
"""

import subprocess
import sys


def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: command failed (exit {result.returncode})")
        sys.exit(1)


if __name__ == "__main__":
    python = sys.executable

    print("Installing bitsandbytes (required for 4-bit QLoRA)...")
    run(f"{python} -m pip install bitsandbytes>=0.43.0")

    print("\nVerifying bitsandbytes CUDA support...")
    run(f"{python} -c \"import bitsandbytes as bnb; print('bitsandbytes:', bnb.__version__)\"")

    print("\nLogging in to HuggingFace (needed to download LLaMA-2)...")
    print("You will need your HuggingFace token (from https://huggingface.co/settings/tokens)")
    print("Also make sure you have accepted the LLaMA-2 license at:")
    print("  https://huggingface.co/meta-llama/Llama-2-7b-hf")
    run("huggingface-cli login")

    print("\nVerifying LLaMA-2 access (downloads tokenizer only — ~500 KB)...")
    run(f"{python} -c \"from transformers import AutoTokenizer; "
        f"AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_fast=False); "
        f"print('Access confirmed.')\"")

    print("\n✓ Setup complete. You can now run:")
    print("  python -m baselines.llama_experiments.run_experiments --dry_run")
    print("  python -m baselines.llama_experiments.run_experiments")
