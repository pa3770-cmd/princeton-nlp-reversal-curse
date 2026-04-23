"""
LLaMA-2 7B QLoRA fine-tuning for Experiments 1 and 3.

Runs 4 fine-tune jobs overnight in sequence:
  Exp 1 — d2n (Description→Name):  train, eval forward + reverse
  Exp 1 — n2d (Name→Description):  train, eval forward + reverse
  Exp 3 — q2a (QuestionToAnswer):   train, eval on held-out Q&A
  Exp 3 — a2q (AnswerToQuestion):   train, eval on held-out Q&A

Resume:  if a run already has results/llama_experiments/<run_id>/results.json
         it is skipped automatically.

Usage (from the code/ directory, MachineLearning env active):
    huggingface-cli login          # one-time, accept Meta LLaMA-2 license first
    python -m baselines.llama_experiments.run_experiments

Options:
    --exp1_epochs   INT   epochs for Experiment 1 runs  (default 50)
    --exp3_epochs   INT   epochs for Experiment 3 runs  (default 20)
    --seed          INT   single training seed           (default 42)
    --skip_exp1         skip Experiment 1 runs
    --skip_exp3         skip Experiment 3 runs
    --dry_run           load data + model but train 1 step only (smoke test)
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from baselines.llama_experiments.data_utils import (
    CompletionDataset, collate_fn, load_exp1, load_exp3,
)
from baselines.llama_experiments.evaluator import evaluate
from baselines.llama_experiments.qlora_setup import free_model, load_model_and_tokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_BASE = Path("results/llama_experiments")


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    run_id:     str
    experiment: str          # "exp1" | "exp3"
    condition:  str          # "d2n"|"n2d" for exp1; "q2a"|"a2q" for exp3
    seed:       int
    epochs:     int
    lr:         float = 2e-4
    batch_size: int   = 1
    grad_accum: int   = 8    # effective batch = batch_size × grad_accum
    max_length: int   = 256
    warmup_ratio: float = 0.05


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    logger = logging.getLogger(str(run_dir))
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                                datefmt="%H:%M:%S")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_run(
    cfg: RunConfig,
    run_dir: Path,
    dry_run: bool,
) -> dict:
    """
    Fine-tune LLaMA-2-7B with QLoRA for one (experiment, condition, seed).
    Saves adapter weights and intermediate loss log to run_dir.
    Returns results dict.
    """
    logger = setup_logging(run_dir)
    logger.info(f"{'='*60}")
    logger.info(f"RUN: {cfg.run_id}")
    logger.info(f"  experiment={cfg.experiment}  condition={cfg.condition}  seed={cfg.seed}")
    logger.info(f"  epochs={cfg.epochs}  lr={cfg.lr}  grad_accum={cfg.grad_accum}")
    logger.info(f"{'='*60}")

    torch.manual_seed(cfg.seed)

    # ---- Load data --------------------------------------------------------
    if cfg.experiment == "exp1":
        splits = load_exp1(cfg.condition)
        train_pairs   = splits["train"]
        eval_pairs    = {"forward": splits["forward"], "reverse": splits["reverse"]}
    else:
        splits = load_exp3(cfg.condition)
        train_pairs   = splits["train"]
        eval_pairs    = {"test": splits["test"]}

    logger.info(f"Train examples: {len(train_pairs)}")
    for k, v in eval_pairs.items():
        logger.info(f"  Eval [{k}]: {len(v)} examples")

    # ---- Load model -------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(logger)

    train_ds = CompletionDataset(train_pairs, tokenizer, max_length=cfg.max_length)
    loader   = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=True, collate_fn=collate_fn,
        pin_memory=False,
    )
    logger.info(f"Dataset tokenized: {len(train_ds)} examples")

    # ---- Optimizer + scheduler --------------------------------------------
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=0.0,
    )

    total_steps   = (len(loader) // cfg.grad_accum) * cfg.epochs
    warmup_steps  = max(1, int(total_steps * cfg.warmup_ratio))
    scheduler     = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info(f"Optimizer steps: {total_steps}  warmup: {warmup_steps}")

    # ---- Training ---------------------------------------------------------
    model.train()
    loss_log   = []   # [(epoch, step, loss), ...]
    global_step = 0
    optimizer.zero_grad()
    t0 = time.time()

    actual_epochs = 1 if dry_run else cfg.epochs

    for epoch in range(1, actual_epochs + 1):
        epoch_loss = 0.0
        n_batches  = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{actual_epochs}", leave=False)
        for step, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(model.device)
            labels         = batch["labels"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / cfg.grad_accum
            loss.backward()

            epoch_loss += loss.item() * cfg.grad_accum
            n_batches  += 1

            if (step + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                pbar.set_postfix(loss=f"{epoch_loss / n_batches:.3f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if dry_run:
                break  # one step only

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_log.append({"epoch": epoch, "loss": round(avg_loss, 4)})
        elapsed  = (time.time() - t0) / 60
        logger.info(f"Epoch {epoch:3d}/{actual_epochs}  loss={avg_loss:.4f}  "
                    f"elapsed={elapsed:.1f}min")

        # Save loss log after every epoch so progress is not lost on crash
        with open(run_dir / "loss_log.json", "w") as f:
            json.dump(loss_log, f, indent=2)

    # ---- Save LoRA adapter ------------------------------------------------
    adapter_dir = run_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info(f"Adapter saved → {adapter_dir}")

    # ---- Evaluation -------------------------------------------------------
    model.eval()
    eval_results = {}

    for split_name, pairs in eval_pairs.items():
        if dry_run:
            pairs = pairs[:4]  # tiny subset for smoke test
        res = evaluate(model, tokenizer, pairs, desc=f"Eval [{split_name}]")
        eval_results[split_name] = res
        logger.info(
            f"  [{split_name}] accuracy={res['accuracy']:.4f}  "
            f"({res['n_correct']}/{res['n_total']})"
        )

    # ---- Save results -----------------------------------------------------
    results = {
        "run_id":     cfg.run_id,
        "experiment": cfg.experiment,
        "condition":  cfg.condition,
        "seed":       cfg.seed,
        "epochs":     actual_epochs,
        "lr":         cfg.lr,
        "dry_run":    dry_run,
        "loss_log":   loss_log,
        "eval":       {k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
                       for k, v in eval_results.items()},
        "timestamp":  datetime.now().isoformat(),
        "train_minutes": round((time.time() - t0) / 60, 1),
    }

    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {results_path}")

    # ---- Release GPU memory before next run --------------------------------
    free_model(model)

    return results


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def build_run_list(args) -> list[RunConfig]:
    runs = []

    if not args.skip_exp1:
        for cond in ["d2n", "n2d"]:
            runs.append(RunConfig(
                run_id     = f"exp1_{cond}_seed{args.seed}",
                experiment = "exp1",
                condition  = cond,
                seed       = args.seed,
                epochs     = args.exp1_epochs,
            ))

    if not args.skip_exp3:
        for cond in ["q2a", "a2q"]:
            runs.append(RunConfig(
                run_id     = f"exp3_{cond}_seed{args.seed}",
                experiment = "exp3",
                condition  = cond,
                seed       = args.seed,
                epochs     = args.exp3_epochs,
            ))

    return runs


def main():
    parser = argparse.ArgumentParser(description="LLaMA-2 QLoRA for Reversal Curse Experiments 1 & 3")
    parser.add_argument("--exp1_epochs", type=int, default=50)
    parser.add_argument("--exp3_epochs", type=int, default=20)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--skip_exp1",   action="store_true")
    parser.add_argument("--skip_exp3",   action="store_true")
    parser.add_argument("--dry_run",     action="store_true",
                        help="1 training step + 4 eval examples — smoke test only")
    args = parser.parse_args()

    RESULTS_BASE.mkdir(parents=True, exist_ok=True)

    # Master log (shared across all runs)
    master_log = RESULTS_BASE / "master_run.log"
    root_logger = logging.getLogger("master")
    root_logger.setLevel(logging.INFO)
    if not root_logger.handlers:
        fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh  = logging.FileHandler(master_log, encoding="utf-8")
        fh.setFormatter(fmt)
        ch  = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        root_logger.addHandler(fh)
        root_logger.addHandler(ch)

    runs = build_run_list(args)
    root_logger.info(f"Scheduled {len(runs)} run(s): {[r.run_id for r in runs]}")
    root_logger.info(f"Dry run: {args.dry_run}")

    summary = []

    for i, cfg in enumerate(runs, 1):
        run_dir = RESULTS_BASE / cfg.run_id
        results_path = run_dir / "results.json"

        # Resume: skip completed runs
        if results_path.exists():
            root_logger.info(f"[{i}/{len(runs)}] SKIP {cfg.run_id} — results already exist")
            with open(results_path) as f:
                summary.append(json.load(f))
            continue

        root_logger.info(f"[{i}/{len(runs)}] START {cfg.run_id}")
        t_start = time.time()

        try:
            results = train_one_run(cfg, run_dir, args.dry_run)
            summary.append(results)
            root_logger.info(
                f"[{i}/{len(runs)}] DONE  {cfg.run_id}  "
                f"({(time.time()-t_start)/60:.1f} min)"
            )
        except Exception as e:
            root_logger.error(f"[{i}/{len(runs)}] FAILED {cfg.run_id}: {e}", exc_info=True)
            # Free GPU even on failure so next run can start
            torch.cuda.empty_cache()
            root_logger.info("GPU cache cleared — continuing with next run")

    # ---- Print final summary table ----------------------------------------
    root_logger.info("\n" + "="*60)
    root_logger.info("FINAL SUMMARY")
    root_logger.info("="*60)
    for r in summary:
        root_logger.info(f"  {r['run_id']}")
        for split, metrics in r.get("eval", {}).items():
            root_logger.info(f"    [{split}] accuracy={metrics.get('accuracy', 'N/A'):.4f}")
    root_logger.info("="*60)

    # Save combined summary
    summary_path = RESULTS_BASE / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    root_logger.info(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
