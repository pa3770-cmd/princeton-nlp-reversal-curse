"""Command-line entry point and run scheduler."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Ensure stdout can carry unicode on Windows (cp1252 chokes on special chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from . import paths
from .config import DEFAULT_EPOCHS, LORA_RANK, RunConfig
from .experiments import MODELS, REGISTRY
from .logging_utils import get_master_logger
from .reevaluation import reevaluate_run
from .training import train_run


# ---------------------------------------------------------------------------
# Run selection
# ---------------------------------------------------------------------------
def parse_only(only: list[str] | None) -> list[tuple[str, str]]:
    """Convert --only tokens into (experiment, condition) pairs.

    Token forms:
      "exp1"        -> all conditions of exp1
      "exp1.d2p"    -> just the d2p condition
    Default (only=None): every (experiment, condition) in the registry.
    """
    if not only:
        return [(exp.name, cond) for exp in REGISTRY.values() for cond in exp.conditions]

    selected: list[tuple[str, str]] = []
    for token in only:
        exp_name, _, cond = token.partition(".")
        if exp_name not in REGISTRY:
            raise SystemExit(f"--only: unknown experiment {exp_name!r}. "
                             f"Choices: {sorted(REGISTRY)}")
        spec = REGISTRY[exp_name]
        if cond:
            if cond not in spec.conditions:
                raise SystemExit(f"--only: unknown condition {cond!r} for {exp_name}. "
                                 f"Choices: {spec.conditions}")
            selected.append((exp_name, cond))
        else:
            selected.extend((exp_name, c) for c in spec.conditions)
    return selected


def build_runs(args) -> list[RunConfig]:
    pairs = parse_only(args.only)
    runs  = []
    for model_tag in args.models:
        model_id = MODELS[model_tag]
        for exp_name, cond in pairs:
            spec = REGISTRY[exp_name]
            runs.append(RunConfig(
                run_id      = f"{model_tag}_{exp_name}_{cond}_seed{args.seed}",
                experiment  = exp_name,
                condition   = cond,
                model_tag   = model_tag,
                model_id    = model_id,
                seed        = args.seed,
                epochs      = args.epochs,
                lr          = args.lr if args.lr is not None else spec.lr_for(cond),
                lora_rank   = args.lora_rank,
                resume_from = args.resume_from,
            ))

    if args.resume_from and len(runs) > 1:
        raise SystemExit("--resume_from only valid when exactly one run is selected "
                         f"(got {len(runs)}). Narrow with --only and --models.")
    return runs


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
async def _run_all(args, master_logger) -> list[dict]:
    runs    = build_runs(args)
    summary = []
    master_logger.info(f"Scheduled {len(runs)} run(s): {[r.run_id for r in runs]}")
    master_logger.info(f"Mode: {'reeval' if args.reeval else 'train'}  dry_run: {args.dry_run}")

    for i, cfg in enumerate(runs, 1):
        run_dir      = paths.RESULTS_DIR / cfg.run_id
        results_path = run_dir / "results.json"

        # Train mode + already done => skip (reeval mode never skips)
        if not args.reeval and results_path.exists():
            master_logger.info(f"[{i}/{len(runs)}] SKIP {cfg.run_id} — already done")
            with open(results_path) as f:
                summary.append(json.load(f))
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        action = "REEVAL" if args.reeval else "START"
        master_logger.info(f"[{i}/{len(runs)}] {action} {cfg.run_id}")
        t0 = time.time()

        try:
            if args.reeval:
                results = await reevaluate_run(
                    run_id=cfg.run_id, run_dir=run_dir,
                    experiment=cfg.experiment, condition=cfg.condition,
                    model_id=cfg.model_id, dry_run=args.dry_run,
                )
            else:
                results = await train_run(cfg, run_dir, args.dry_run)
            summary.append(results)
            master_logger.info(
                f"[{i}/{len(runs)}] DONE  {cfg.run_id}  ({(time.time() - t0) / 60:.1f} min)"
            )
        except Exception as e:
            master_logger.error(f"[{i}/{len(runs)}] FAILED {cfg.run_id}: {e}", exc_info=True)
            master_logger.info("Continuing with next run.")

    return summary


def _print_summary(summary: list[dict], logger) -> None:
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    for r in summary:
        logger.info(f"  {r['run_id']}  ({r.get('model', '?')})")
        for split, metrics in r.get("eval", {}).items():
            acc = metrics.get("accuracy", "N/A")
            logger.info(f"    [{split}] accuracy={acc:.4f}" if isinstance(acc, float)
                        else f"    [{split}] accuracy={acc}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tinker LoRA fine-tuning for Reversal Curse experiments")
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                   choices=list(MODELS.keys()),
                   help="Models to run (default: all registered)")
    p.add_argument("--only", nargs="+", default=None, metavar="EXP[.COND]",
                   help='Filter runs. e.g. "exp1" or "exp1.d2p exp3.same" (default: all)')
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                   help=f"Training epochs (default: {DEFAULT_EPOCHS})")
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate override (default: per-condition from ExperimentSpec)")
    p.add_argument("--lora_rank", type=int, default=LORA_RANK,
                   help=f"LoRA rank (default: {LORA_RANK})")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_from", type=str, default=None,
                   help="Tinker checkpoint path; only valid when one run is selected")
    p.add_argument("--reeval", action="store_true",
                   help="Re-run eval on already-trained runs using saved weights")
    p.add_argument("--dry_run", action="store_true",
                   help="1 training step + 4 eval examples — smoke test")
    return p


async def main_async(args) -> None:
    paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    master_logger = get_master_logger(paths.RESULTS_DIR)

    summary = await _run_all(args, master_logger)
    _print_summary(summary, master_logger)

    summary_path = paths.RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    master_logger.info(f"Summary saved -> {summary_path}")


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
