#!/usr/bin/env python3
"""
finetune.py — Fine-tune any causal LM on a JSONL prompt-completion dataset
and evaluate forward and reverse accuracy.

This is the core script for all fine-tuning experiments in the Reversal Curse
project. Every experiment (original baselines, SmolLM2, bidirectional, etc.)
runs through this script with different --model / --train / --test_* flags.

Results are saved to:
    results/{experiment}/{model_slug}_seed{seed}.json
Checkpoints (optional) are saved to:
    checkpoints/{experiment}/{model_slug}_seed{seed}/

Usage examples
--------------
# Original GPT-2 Small baseline (description->name direction)
python finetune.py \\
  --model gpt2 \\
  --train  ../original_repo/data/reverse_experiments/june_version_7921032488/d2p_prompts_train.jsonl \\
  --test_forward ../original_repo/data/reverse_experiments/june_version_7921032488/d2p_prompts_test.jsonl \\
  --test_reverse ../original_repo/data/reverse_experiments/june_version_7921032488/d2p_reverse_prompts_test.jsonl \\
  --epochs 100 --seed 42 --experiment original_baseline --eval_every 10

# GPT-2 Medium, seed 1
python finetune.py --model gpt2-medium --epochs 100 --seed 1 --experiment original_baseline \\
  --train  ../original_repo/data/reverse_experiments/june_version_7921032488/d2p_prompts_train.jsonl \\
  --test_forward ../original_repo/data/reverse_experiments/june_version_7921032488/d2p_prompts_test.jsonl \\
  --test_reverse ../original_repo/data/reverse_experiments/june_version_7921032488/d2p_reverse_prompts_test.jsonl

# SmolLM2-135M
python finetune.py --model HuggingFaceTB/SmolLM2-135M --epochs 100 --seed 42 \\
  --experiment smollm2_baseline \\
  --train  ../original_repo/data/reverse_experiments/june_version_7921032488/d2p_prompts_train.jsonl \\
  --test_forward ../original_repo/data/reverse_experiments/june_version_7921032488/d2p_prompts_test.jsonl \\
  --test_reverse ../original_repo/data/reverse_experiments/june_version_7921032488/d2p_reverse_prompts_test.jsonl

# Multi-hop (synthetic data)
python finetune.py --model gpt2-medium --epochs 100 --seed 42 \\
  --experiment multihop \\
  --train data/multihop/train.jsonl \\
  --test_forward data/multihop/test_2hop_forward.jsonl data/multihop/test_3hop_forward.jsonl \\
  --test_reverse data/multihop/test_2hop_reverse.jsonl data/multihop/test_3hop_reverse.jsonl
"""

import argparse
import json
import os
import random
import sys
import time
import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

# Allow running from either the code/ directory or the repo root
sys.path.insert(0, str(Path(__file__).parent))
from training.data_utils   import load_jsonl, CompletionDataset
from training.eval_utils   import evaluate_prefix_match, evaluate_log_prob
from training.results_utils import save_run


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log(log_path: str, message: str):
    """Append a timestamped line to the run log file."""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {message}"
    print(line, flush=True)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    epochs: int,
    use_fp16: bool,
    throttle_sleep: float = 0.0,
    eval_every: int = 0,
    eval_callback=None,
    loss_label: str = "loss",
    log_path: str = "",
) -> list[float]:
    """
    Run the training loop and return per-epoch mean loss values.

    Uses torch.cuda.amp for mixed-precision when use_fp16=True (recommended
    on RTX 3060 for ~1.5x speedup with no accuracy impact on fine-tuning).

    throttle_sleep: seconds to sleep after each batch to reduce GPU utilization.
                    0.025 keeps the GPU at ~80% on RTX 3060.
    eval_every:     if > 0, call eval_callback() every N epochs and log accuracy.
                    Use 1 to log fwd_acc/rev_acc after every epoch.
    eval_callback:  callable that returns (fwd_sw, rev_sw, fwd_lp, rev_lp).
    loss_label:     label used in epoch log lines, e.g. "fs_loss" or "cl_loss".
    log_path:       if set, all log lines are also written to this file.
    """
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    model.train()
    epoch_losses: list[float] = []
    t_run_start = time.time()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches  = 0
        t_epoch_start = time.time()

        for batch_idx, batch in enumerate(dataloader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_fp16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            n_batches  += 1

            if (batch_idx + 1) % 100 == 0:
                batch_msg = (
                    f"  BATCH  epoch={epoch:>3}  batch={batch_idx+1:>4}/{len(dataloader)}"
                    f"  loss={loss.item():.4f}"
                )
                if log_path:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {batch_msg}\n")
                print(batch_msg, flush=True)

            if throttle_sleep > 0.0:
                time.sleep(throttle_sleep)

        epoch_loss = total_loss / n_batches if n_batches else 0.0
        epoch_losses.append(round(epoch_loss, 6))

        elapsed    = time.time() - t_run_start
        epoch_time = time.time() - t_epoch_start
        remaining  = elapsed / epoch * (epochs - epoch) if epoch > 0 else 0
        eta        = datetime.datetime.now() + datetime.timedelta(seconds=remaining)

        # Run accuracy eval if due
        fwd_sw = rev_sw = fwd_lp = rev_lp = None
        if eval_every > 0 and epoch % eval_every == 0 and eval_callback is not None:
            fwd_sw, rev_sw, fwd_lp, rev_lp = eval_callback()
            model.train()  # restore train mode after eval

        # Build epoch log line — always includes loss; includes acc if eval ran
        epoch_msg = (
            f"  EPOCH {epoch:>3}/{epochs}"
            f"  {loss_label}={epoch_loss:.4f}"
            f"  epoch_time={epoch_time:.1f}s"
            f"  elapsed={elapsed/60:.1f}min"
            f"  ETA={eta.strftime('%H:%M:%S')}"
        )
        if fwd_sw is not None:
            epoch_msg += (
                f"  fwd_acc={fwd_sw:.4f}"
                f"  rev_acc={rev_sw:.4f}"
                f"  fwd_lp={fwd_lp:.4f}"
                f"  rev_lp={rev_lp:.4f}"
            )

        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}]{epoch_msg}", flush=True)
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}]{epoch_msg}\n")

    return epoch_losses


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_file(
    model,
    tokenizer,
    path: Path,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
) -> dict:
    """
    Evaluate a single test file using both prefix_match and log_prob.
    Returns a dict with both metric results plus the file path.
    """
    records = load_jsonl(path)
    print(f"    Evaluating {path.name}  ({len(records)} records)")

    prefix = evaluate_prefix_match(
        model, tokenizer, records, device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    logprob = evaluate_log_prob(
        model, tokenizer, records, device,
    )

    sw  = prefix["startswith_acc"]   # paper-matching metric
    p8  = prefix["prefix_8_acc"]     # lenient 8-word metric
    lp  = logprob["mean_log_prob"]
    print(f"      startswith_acc={sw:.4f}  prefix_8_acc={p8:.4f}  mean_log_prob={lp:.4f}")

    return {
        "file":            str(path),
        "n":               prefix["n"],
        # Primary metric -- matches Berglund et al. 2023
        "startswith_acc":  round(sw, 6),
        # Secondary metrics
        "prefix_8_acc":    round(p8, 6),
        "mean_log_prob":   round(lp, 6),
        "predictions":     prefix["predictions"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune a causal LM on prompt-completion JSONL and evaluate "
                    "forward / reverse accuracy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--model", default="gpt2",
                   help="HuggingFace model name or local path")

    # Data
    p.add_argument("--train", required=True,
                   help="Path to training JSONL")
    p.add_argument("--test_forward", nargs="+", default=[],
                   help="One or more forward-direction test JSONL paths")
    p.add_argument("--test_reverse", nargs="+", default=[],
                   help="One or more reverse-direction test JSONL paths")

    # Training hyper-parameters
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=5e-5)
    p.add_argument("--warmup_steps", type=int, default=50,
                   help="Linear LR warmup steps")
    p.add_argument("--max_length", type=int,   default=256,
                   help="Max token length for training sequences")
    p.add_argument("--completion_only_loss", action="store_true", default=False,
                   help="Mask prompt tokens from loss (NOT paper-matching). "
                        "Default is full-sequence loss to match Berglund et al. 2023.")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--fp16",       action="store_true", default=True,
                   help="Use mixed-precision training (recommended on RTX 3060)")
    p.add_argument("--no_fp16",    dest="fp16", action="store_false")
    p.add_argument("--throttle_sleep", type=float, default=0.0,
                   help="Seconds to sleep after each batch to reduce GPU utilization. "
                        "0.025 keeps the GPU at ~80%% on RTX 3060.")

    # Evaluation
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens",  type=int, default=50,
                   help="Generation budget for prefix_match evaluation")
    p.add_argument("--eval_every", type=int, default=0,
                   help="Run forward/reverse accuracy snapshot every N epochs during "
                        "training (0 = only at the end). Recommended: 10 for 100-epoch runs.")

    # Output
    p.add_argument("--experiment",   required=True,
                   help="Experiment name (used as sub-directory under results/ and checkpoints/)")
    p.add_argument("--results_dir",  default="results",
                   help="Root directory for result JSON files")
    p.add_argument("--log_dir",      default="logs",
                   help="Directory for per-run log files (one .log file per run)")
    p.add_argument("--save_model",   action="store_true", default=False,
                   help="Save the fine-tuned model checkpoint to checkpoints/")
    p.add_argument("--checkpoints_dir", default="checkpoints",
                   help="Root directory for model checkpoints (only used if --save_model)")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Set up log file
    # ------------------------------------------------------------------
    model_slug = args.model.replace("/", "--")
    log_dir    = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path   = str(log_dir / f"{args.experiment}_{model_slug}_seed{args.seed}.log")
    # Write header
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"model={args.model}  experiment={args.experiment}  seed={args.seed}\n")
        f.write(f"epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}"
                f"  fp16={args.fp16 and device.type == 'cuda'}\n")
        f.write("=" * 72 + "\n")

    print(f"\n{'='*60}")
    print(f"  Model:      {args.model}")
    print(f"  Experiment: {args.experiment}")
    print(f"  Seed:       {args.seed}")
    print(f"  Device:     {device}")
    if device.type == "cuda":
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
    print(f"  FP16:       {args.fp16 and device.type == 'cuda'}")
    print(f"  Log file:   {log_path}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Load tokenizer & model
    # ------------------------------------------------------------------
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # GPT-2 and similar models have no dedicated pad token — reuse EOS
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {n_params:.1f}M parameters\n")

    # ------------------------------------------------------------------
    # Build training dataset & dataloader
    # ------------------------------------------------------------------
    print(f"Loading training data: {args.train}")
    train_records = load_jsonl(args.train)
    print(f"  {len(train_records)} training examples\n")

    full_seq_loss = not args.completion_only_loss
    print(f"  Loss mode:  {'full-sequence (prompt+completion)' if full_seq_loss else 'completion-only'}\n")
    train_dataset = CompletionDataset(train_records, tokenizer, max_length=args.max_length,
                                      full_sequence_loss=full_seq_loss)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,          # Windows-safe default
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # ------------------------------------------------------------------
    # Build eval callback for mid-training accuracy snapshots
    # ------------------------------------------------------------------
    eval_callback = None
    if args.eval_every > 0 and (args.test_forward or args.test_reverse):
        fwd_records = load_jsonl(args.test_forward[0]) if args.test_forward else []
        rev_records = load_jsonl(args.test_reverse[0]) if args.test_reverse else []

        def eval_callback():
            fwd_sw = rev_sw = fwd_lp = rev_lp = 0.0
            if fwd_records:
                r = evaluate_prefix_match(model, tokenizer, fwd_records, device,
                                          batch_size=args.eval_batch_size,
                                          max_new_tokens=args.max_new_tokens)
                lp = evaluate_log_prob(model, tokenizer, fwd_records, device)
                fwd_sw = r["startswith_acc"]
                fwd_lp = lp["mean_log_prob"]
            if rev_records:
                r = evaluate_prefix_match(model, tokenizer, rev_records, device,
                                          batch_size=args.eval_batch_size,
                                          max_new_tokens=args.max_new_tokens)
                lp = evaluate_log_prob(model, tokenizer, rev_records, device)
                rev_sw = r["startswith_acc"]
                rev_lp = lp["mean_log_prob"]
            return fwd_sw, rev_sw, fwd_lp, rev_lp

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"Training for {args.epochs} epochs  "
          f"({len(train_loader)} batches/epoch, {total_steps} total steps)...")
    if args.eval_every > 0:
        print(f"  Accuracy snapshots every {args.eval_every} epochs\n")
    t0 = time.time()
    use_fp16 = args.fp16 and device.type == "cuda"
    loss_label = "fs_loss" if full_seq_loss else "cl_loss"
    # Always eval every epoch when a log file is present so every epoch line
    # gets fwd_acc/rev_acc; fall back to args.eval_every if no callback built.
    effective_eval_every = 1 if (eval_callback is not None) else args.eval_every
    epoch_losses = train(
        model, train_loader, optimizer, scheduler, device, args.epochs, use_fp16,
        throttle_sleep=args.throttle_sleep,
        eval_every=effective_eval_every,
        eval_callback=eval_callback,
        loss_label=loss_label,
        log_path=log_path,
    )
    train_time   = time.time() - t0
    print(f"\nTraining complete in {train_time/60:.1f} min  "
          f"(final loss: {epoch_losses[-1]:.4f})\n")

    # ------------------------------------------------------------------
    # Final evaluation on all test files
    # ------------------------------------------------------------------
    test_results: dict[str, dict] = {}

    fwd_sw:  list[float] = []
    rev_sw:  list[float] = []
    fwd_p8:  list[float] = []
    rev_p8:  list[float] = []
    fwd_lps: list[float] = []
    rev_lps: list[float] = []

    def run_eval(paths: list[str], sw_col: list, p8_col: list, lp_col: list):
        for path_str in paths:
            path = Path(path_str)
            if not path.exists():
                print(f"  WARNING: test file not found - {path}")
                continue
            result = evaluate_file(model, tokenizer, path, device,
                                   args.eval_batch_size, args.max_new_tokens)
            test_results[path.stem] = result
            sw_col.append(result["startswith_acc"])
            p8_col.append(result["prefix_8_acc"])
            lp_col.append(result["mean_log_prob"])

    if args.test_forward:
        print("Forward tests:")
        run_eval(args.test_forward, fwd_sw, fwd_p8, fwd_lps)

    if args.test_reverse:
        print("Reverse tests:")
        run_eval(args.test_reverse, rev_sw, rev_p8, rev_lps)

    def _mean(lst): return sum(lst) / len(lst) if lst else None

    forward_acc    = _mean(fwd_sw)
    reverse_acc    = _mean(rev_sw)
    forward_p8     = _mean(fwd_p8)
    reverse_p8     = _mean(rev_p8)
    forward_lp     = _mean(fwd_lps)
    reverse_lp     = _mean(rev_lps)

    print(f"\n{'='*60}")
    print(f"  Metric comparison against Berglund et al. 2023")
    print(f"  (paper expects: forward ~96%, reverse ~3%)")
    print(f"  {'Metric':<25} {'Forward':>10} {'Reverse':>10}")
    print(f"  {'-'*45}")
    if forward_acc is not None:
        print(f"  {'startswith_acc (paper)':<25} {forward_acc:>10.4f} {(reverse_acc or 0):>10.4f}")
    if forward_p8 is not None:
        print(f"  {'prefix_8_acc (lenient)':<25} {forward_p8:>10.4f} {(reverse_p8 or 0):>10.4f}")
    if forward_lp is not None:
        print(f"  {'mean_log_prob':<25} {forward_lp:>10.4f} {(reverse_lp or 0):>10.4f}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    metrics = {
        "forward_acc":      forward_acc,
        "reverse_acc":      reverse_acc,
        "forward_p8_acc":   forward_p8,
        "reverse_p8_acc":   reverse_p8,
        "forward_log_prob": forward_lp,
        "reverse_log_prob": reverse_lp,
        "training": {
            "epochs":          args.epochs,
            "batch_size":      args.batch_size,
            "lr":              args.lr,
            "warmup_steps":    args.warmup_steps,
            "fp16":            use_fp16,
            "full_sequence_loss": full_seq_loss,
            "train_file":      str(Path(args.train).resolve()),
            "n_train":         len(train_records),
            "train_time_min":  round(train_time / 60, 2),
            "loss_per_epoch":  epoch_losses,
            "final_loss":      epoch_losses[-1],
        },
        "test_results": test_results,
    }

    results_path = save_run(
        results_dir=args.results_dir,
        experiment=args.experiment,
        model_name=args.model,
        seed=args.seed,
        metrics=metrics,
    )
    print(f"Results saved -> {results_path}")

    if args.save_model:
        ckpt_dir     = Path(args.checkpoints_dir) / args.experiment / f"{model_slug}_seed{args.seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"Checkpoint saved -> {ckpt_dir}")


if __name__ == "__main__":
    main()
