"""Train one run on Tinker. Single linear path — no eval-only branching."""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

import tinker
from tinker import types

from .config import (
    CHECKPOINT_EVERY, EARLY_STOP_LOSS, EARLY_STOP_PATIENCE, RunConfig,
)
from .data import load_splits
from .evaluation import evaluate_split
from .experiments import REGISTRY
from .logging_utils import get_run_logger
from .tokenization import make_datum


def _write_json(path: Path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


async def _run_probe(s_client, tokenizer, run_id: str, train_pairs, forward_pairs) -> None:
    """Print sample model generations on train + held-out forward examples."""
    params = types.SamplingParams(max_tokens=30, temperature=0.0, stop=["\n"])
    for label, probe_pairs in [("TRAIN", train_pairs[:5]), ("TEST-forward", forward_pairs[:5])]:
        print(f"\n{'=' * 60}\n  PROBE [{run_id}] — {label}\n{'=' * 60}")
        for idx, ex in enumerate(probe_pairs, 1):
            inp    = types.ModelInput.from_ints(tokens=tokenizer.encode(ex["prompt"]))
            result = await s_client.sample_async(prompt=inp, num_samples=1, sampling_params=params)
            got = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True).strip()
            print(f"\n  [{idx}] PROMPT:   {ex['prompt'][:100]}")
            print(f"       EXPECTED: {ex['completion'].strip()!r}")
            print(f"       GOT:      {got!r}")
    print()


async def _train_loop(t_client, data, cfg: RunConfig, run_dir: Path, dry_run: bool, logger):
    """Run gradient steps; return (loss_log, sampling_path).

    sampling_path is the tinker:// URI of the final saved weights — written to results.json
    so reevaluation doesn't need to scan checkpoint logs.
    """
    loss_log         = []
    n_optim_steps    = 0
    epochs_below_thr = 0
    last_state_path  = None
    actual_epochs    = 1 if dry_run else cfg.epochs
    t0               = time.time()

    for epoch in range(1, actual_epochs + 1):
        epoch_loss = 0.0
        n_batches  = 0
        batch_acc  = []

        for i, datum in enumerate(data):
            batch_acc.append(datum)
            if len(batch_acc) == cfg.batch_size or i == len(data) - 1:
                fwd = await (await t_client.forward_backward_async(
                    data=batch_acc, loss_fn="cross_entropy",
                )).result_async()
                epoch_loss += fwd.metrics.get("loss:sum", 0.0)
                n_batches  += 1
                batch_acc   = []

                if n_batches % cfg.grad_accum == 0:
                    await (await t_client.optim_step_async(
                        types.AdamParams(learning_rate=cfg.lr),
                    )).result_async()
                    n_optim_steps += 1

            if dry_run and n_batches >= 1:
                break

        # Flush trailing accumulated grads
        if n_batches % cfg.grad_accum != 0:
            await (await t_client.optim_step_async(
                types.AdamParams(learning_rate=cfg.lr),
            )).result_async()
            n_optim_steps += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed  = (time.time() - t0) / 60
        entry    = {"epoch": epoch, "loss": round(avg_loss, 4)}
        loss_log.append(entry)
        logger.info(
            f"Epoch {epoch:3d}/{actual_epochs}  loss={avg_loss:.4f}  "
            f"steps={n_optim_steps}  elapsed={elapsed:.1f}min"
        )

        epochs_below_thr = epochs_below_thr + 1 if avg_loss < EARLY_STOP_LOSS else 0
        if epochs_below_thr >= EARLY_STOP_PATIENCE:
            logger.info(
                f"  Early stop: loss < {EARLY_STOP_LOSS} for "
                f"{EARLY_STOP_PATIENCE} consecutive epochs."
            )
            _write_json(run_dir / "loss_log.json", loss_log)
            break

        is_checkpoint = (epoch % CHECKPOINT_EVERY == 0) and not dry_run
        is_final      = (epoch == actual_epochs)
        if is_checkpoint and not is_final:
            ckpt_name = f"{cfg.run_id}_epoch{epoch}"
            res = await (await t_client.save_state_async(name=ckpt_name)).result_async()
            entry["checkpoint_path"] = res.path
            last_state_path = res.path
            logger.info(f"  Checkpoint saved: {res.path}")

        _write_json(run_dir / "loss_log.json", loss_log)

    return loss_log, last_state_path, t0


def _derive_sampling_path(state_path: str | None, run_id: str) -> str | None:
    """A state checkpoint path looks like  tinker://<session>:train:N/weights/<ckpt_name>.
    Final weights live at  tinker://<session>:train:N/weights/<run_id>.
    """
    if not state_path:
        return None
    prefix = state_path.rsplit("/", 1)[0]
    return f"{prefix}/{run_id}"


async def train_run(cfg: RunConfig, run_dir: Path, dry_run: bool) -> dict:
    logger = get_run_logger(run_dir)
    spec   = REGISTRY[cfg.experiment]

    logger.info("=" * 60)
    logger.info(f"RUN: {cfg.run_id}")
    logger.info(f"  model={cfg.model_id}  exp={cfg.experiment}  cond={cfg.condition}")
    logger.info(
        f"  epochs={cfg.epochs}  lr={cfg.lr}  batch={cfg.batch_size}  "
        f"accum={cfg.grad_accum}  lora_rank={cfg.lora_rank}"
    )
    logger.info("=" * 60)

    train_pairs, eval_pairs = load_splits(spec, cfg.condition)
    logger.info(
        f"Train: {len(train_pairs)}  |  "
        + "  ".join(f"eval[{k}]: {len(v)}" for k, v in eval_pairs.items())
    )

    service = tinker.ServiceClient()
    if cfg.resume_from:
        logger.info(f"Resuming from checkpoint: {cfg.resume_from}")
        t_client = await service.create_training_client_from_state_with_optimizer_async(
            path=cfg.resume_from,
        )
    else:
        t_client = await service.create_lora_training_client_async(
            base_model=cfg.model_id, rank=cfg.lora_rank,
        )

    tokenizer = t_client.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = [
        make_datum(p["prompt"], p["completion"], tokenizer, cfg.max_length)
        for p in train_pairs
    ]
    logger.info(f"Tokenized {len(data)} training examples")

    loss_log, last_state_path, t0 = await _train_loop(
        t_client, data, cfg, run_dir, dry_run, logger,
    )

    s_client = await t_client.save_weights_and_get_sampling_client_async(name=cfg.run_id)
    logger.info("Final weights saved, sampling client ready.")
    sampling_path = _derive_sampling_path(last_state_path, cfg.run_id)

    if spec.run_probe:
        forward_pairs = next(iter(eval_pairs.values()))
        await _run_probe(s_client, tokenizer, cfg.run_id, train_pairs, forward_pairs)

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
        "run_id":        cfg.run_id,
        "model":         cfg.model_id,
        "experiment":    cfg.experiment,
        "condition":     cfg.condition,
        "seed":          cfg.seed,
        "epochs":        1 if dry_run else cfg.epochs,
        "lr":            cfg.lr,
        "dry_run":       dry_run,
        "sampling_path": sampling_path,
        "loss_log":      loss_log,
        "eval":          {
            k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
            for k, v in eval_results.items()
        },
        "timestamp":     datetime.now().isoformat(),
        "train_minutes": round((time.time() - t0) / 60, 1),
    }
    _write_json(run_dir / "results.json", results)
    logger.info(f"Results saved -> {run_dir / 'results.json'}")
    return results
