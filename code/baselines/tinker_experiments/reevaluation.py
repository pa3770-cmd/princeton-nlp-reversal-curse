"""Re-run evaluation against already-saved Tinker weights.

Reads the sampling path from results.json (preferred) or reconstructs it from
loss_log.json checkpoint entries (fallback for runs done before the path was saved).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import tinker

from .data import load_splits
from .evaluation import evaluate_split
from .experiments import REGISTRY
from .logging_utils import get_run_logger


def _resolve_sampling_path(run_dir: Path, run_id: str) -> str:
    results_path = run_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            saved = json.load(f)
        if saved.get("sampling_path"):
            return saved["sampling_path"]

    loss_log_path = run_dir / "loss_log.json"
    if loss_log_path.exists():
        with open(loss_log_path) as f:
            entries = json.load(f)
        for entry in entries:
            if "checkpoint_path" in entry:
                prefix = entry["checkpoint_path"].rsplit("/", 1)[0]
                return f"{prefix}/{run_id}"

    raise FileNotFoundError(
        f"Cannot resolve Tinker sampling path for '{run_id}'. "
        f"Expected sampling_path in {results_path} or a checkpoint_path in {loss_log_path}."
    )


async def reevaluate_run(
    run_id: str, run_dir: Path, experiment: str, condition: str, model_id: str,
    dry_run: bool = False,
) -> dict:
    logger = get_run_logger(run_dir)
    spec   = REGISTRY[experiment]

    sampling_path = _resolve_sampling_path(run_dir, run_id)
    logger.info(f"Reeval: loading weights from {sampling_path}")

    service   = tinker.ServiceClient()
    s_client  = await service.create_sampling_client_async(model_path=sampling_path)
    tokenizer = s_client.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _, eval_pairs = load_splits(spec, condition)

    eval_results = {}
    for split_name, pairs in eval_pairs.items():
        if dry_run:
            pairs = pairs[:4]
        res = await evaluate_split(
            s_client, tokenizer, pairs,
            desc=f"Eval [{split_name}]", strip_after=spec.eval_strip,
        )
        eval_results[split_name] = res
        logger.info(
            f"  [{split_name}] accuracy={res['accuracy']:.4f}  "
            f"({res['n_correct']}/{res['n_total']})"
        )

    results = {
        "run_id":        run_id,
        "model":         model_id,
        "experiment":    experiment,
        "condition":     condition,
        "eval_only":     True,
        "sampling_path": sampling_path,
        "eval":          {
            k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
            for k, v in eval_results.items()
        },
        "timestamp":     datetime.now().isoformat(),
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved -> {run_dir / 'results.json'}")
    return results
