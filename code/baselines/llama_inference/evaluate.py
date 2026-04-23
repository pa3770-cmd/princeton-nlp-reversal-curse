"""Llama base-model inference baseline on celebrity parent-child reversal task.

Mirrors baselines.celebrity_api.evaluate but routes inference through Tinker
instead of OpenAI. Same scoring rule (any-of-N starts-with), same prompts,
same dataset — drops into results/api_eval/ alongside the GPT runs.

Usage (from Project/code/, with TinkerEnv active and TINKER_API_KEY set):
    python -m baselines.llama_inference.evaluate                         # full 1513 pairs
    python -m baselines.llama_inference.evaluate --max_pairs 3           # smoke test
    python -m baselines.llama_inference.evaluate --model meta-llama/Llama-3.1-8B-Instruct
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

# Stdout unicode for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from baselines.celebrity_api.prompts import build_child_query, build_parent_query
from baselines.celebrity_api.scoring import score_responses
from baselines.llama_inference.cot_prompts import (
    COT_MAX_TOKENS,
    COT_STOP,
    DEMO_PAIRS,
    HINT_MAX_TOKENS,
    HINT_STOP,
    K_DEFAULT,
    build_fewshot_cot_reverse,
    build_hint_reverse,
    build_zeroshot_cot_reverse,
    demo_pairs_for_k,
    is_leaked,
)
from baselines.llama_inference.cot_scoring import extract_answer, score_cot_responses
from baselines.llama_inference.tinker_sampler import make_base_sampling_client, sample

# ---------------------------------------------------------------------------
# Paths — anchored, cwd-independent
# ---------------------------------------------------------------------------
_PKG_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = _PKG_DIR.parents[2]                                # .../Project
DATA_PATH    = PROJECT_ROOT / "original_repo" / "data" / "celebrity_relations" / "parent_child_pairs.csv"
RESULTS_DIR  = PROJECT_ROOT / "code" / "results" / "api_eval"

# ---------------------------------------------------------------------------
# Defaults — match baselines.celebrity_api.evaluate so numbers are comparable
# ---------------------------------------------------------------------------
DEFAULT_MODEL    = "meta-llama/Llama-3.1-70B-Instruct"
N_SAMPLES        = 10
TEMPERATURE      = 1.0
MAX_TOKENS       = 30
STOP             = ["\n"]
CHECKPOINT_EVERY = 50

# Reverse-direction-only conditions. Each maps to (prompt builder, max_tokens, stop).
# All three share the same scorer (cot_scoring.score_cot_responses) and the same
# is_leaked() filter, so they evaluate on the same pair set.
COT_CONDITIONS = {
    "hint":         (build_hint_reverse,         HINT_MAX_TOKENS, HINT_STOP),
    "zeroshot_cot": (build_zeroshot_cot_reverse, COT_MAX_TOKENS,  COT_STOP),
    "fewshot_cot":  (build_fewshot_cot_reverse,  COT_MAX_TOKENS,  COT_STOP),
}


def _safe_name(s: str) -> str:
    """Filesystem tag from a HF model id, e.g. 'meta-llama/Llama-3.1-70B-Instruct' -> 'llama-3.1-70b-instruct'."""
    last = s.rsplit("/", 1)[-1].lower()
    return re.sub(r"[^\w.\-]", "_", last)


async def _evaluate_pair(s_client, tokenizer, child, parent, parent_type, n):
    fwd_msgs = build_parent_query(child, parent_type)
    rev_msgs = build_child_query(parent)
    fwd_resp, rev_resp = await asyncio.gather(
        sample(s_client, tokenizer, fwd_msgs, n=n,
               temperature=TEMPERATURE, max_tokens=MAX_TOKENS, stop=STOP),
        sample(s_client, tokenizer, rev_msgs, n=n,
               temperature=TEMPERATURE, max_tokens=MAX_TOKENS, stop=STOP),
    )
    return (
        score_responses(fwd_resp, parent),
        score_responses(rev_resp, child),
        fwd_resp, rev_resp,
    )


async def run_evaluation(
    df: pd.DataFrame, s_client, tokenizer, tag: str, n_samples: int,
    csv_path: Path, verbose: bool, concurrency: int,
) -> tuple[pd.DataFrame, int]:
    fwd_col = f"{tag}_can_find_parent"
    rev_col = f"{tag}_can_find_child"

    results = df[["child", "parent", "parent_type", "child_prediction"]].copy()
    results[fwd_col] = float("nan")
    results[rev_col] = float("nan")

    # Resume: pick up after the last fully-scored row
    start_idx = 0
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        if fwd_col in existing.columns:
            done = int(existing[fwd_col].notna().sum())
            if done > 0:
                print(f"Resuming from pair {done} / {len(df)}", flush=True)
                results.loc[:done - 1, fwd_col] = existing[fwd_col].iloc[:done].values
                results.loc[:done - 1, rev_col] = existing[rev_col].iloc[:done].values
                start_idx = done

    pending  = df.iloc[start_idx:].reset_index(drop=True)
    n_failed = 0
    n_done   = 0
    t0       = time.time()
    sem      = asyncio.Semaphore(concurrency)

    async def process(i: int, row) -> tuple[int, str, float | None, float | None, list, list, Exception | None]:
        async with sem:
            try:
                fwd_score, rev_score, fwd_resp, rev_resp = await _evaluate_pair(
                    s_client, tokenizer,
                    row["child"], row["parent"], row["parent_type"], n_samples,
                )
                return start_idx + i, row["child"], fwd_score, rev_score, fwd_resp, rev_resp, None
            except Exception as e:
                return start_idx + i, row["child"], None, None, [], [], e

    tasks = [
        asyncio.create_task(process(i, row))
        for i, (_, row) in enumerate(pending.iterrows())
    ]
    print(f"Dispatched {len(tasks)} pairs at concurrency={concurrency}", flush=True)

    for completed in asyncio.as_completed(tasks):
        global_idx, child, fwd_score, rev_score, fwd_resp, rev_resp, err = await completed

        if err is not None:
            print(f"  FAILED pair {global_idx+1}/{len(df)} {child!r}: {type(err).__name__}: {err}", flush=True)
            n_failed += 1
            continue

        results.at[global_idx, fwd_col] = fwd_score
        results.at[global_idx, rev_col] = rev_score
        n_done += 1

        if verbose:
            row = df.iloc[global_idx]
            print(f"  [{global_idx+1}/{len(df)}] {child} <- {row['parent']} ({row['parent_type']}): "
                  f"fwd={fwd_score:.0f} rev={rev_score:.0f}", flush=True)
            print(f"      fwd[0]: {fwd_resp[0]!r}", flush=True)
            print(f"      rev[0]: {rev_resp[0]!r}", flush=True)
        elif n_done % 25 == 0:
            elapsed = (time.time() - t0) / 60
            rate    = n_done / max(elapsed, 1e-6)
            eta     = (len(pending) - n_done) / max(rate, 1e-6)
            print(f"  done={n_done}/{len(pending)}  elapsed={elapsed:.1f}min  "
                  f"rate={rate:.1f}/min  eta={eta:.1f}min  failed={n_failed}", flush=True)

        if n_done % CHECKPOINT_EVERY == 0:
            results.to_csv(csv_path, index=False)

    return results, n_failed


async def _evaluate_pair_cot(s_client, tokenizer, parent, child, n, builder, max_tokens, stop):
    """Reverse-direction only: 'Name a child of {parent}.' with the given builder."""
    rev_msgs = builder(parent)
    rev_resp = await sample(
        s_client, tokenizer, rev_msgs, n=n,
        temperature=TEMPERATURE, max_tokens=max_tokens, stop=stop,
    )
    return score_cot_responses(rev_resp, child), rev_resp


async def run_cot_evaluation(
    df: pd.DataFrame, s_client, tokenizer, tag: str, n_samples: int,
    csv_path: Path, verbose: bool, concurrency: int,
    builder, max_tokens: int, stop: list[str],
) -> tuple[pd.DataFrame, int, int]:
    rev_col      = f"{tag}_can_find_child"
    leaked_col   = f"{tag}_demo_leakage"
    response_col = f"{tag}_first_response"

    # Pre-filter pairs whose names overlap with the few-shot CoT demos.
    flags = df.apply(lambda r: is_leaked(r["parent"], r["child"]), axis=1)
    n_leaked = int(flags.sum())
    if n_leaked:
        print(f"Filtering {n_leaked} pairs that overlap with CoT demos: "
              f"{df[flags][['child','parent']].values.tolist()}", flush=True)

    results = df[["child", "parent", "parent_type", "child_prediction"]].copy()
    results[rev_col]      = float("nan")
    results[leaked_col]   = flags.values
    results[response_col] = ""

    pending  = df[~flags].reset_index().rename(columns={"index": "global_idx"})
    n_failed = 0
    n_done   = 0
    t0       = time.time()
    sem      = asyncio.Semaphore(concurrency)

    async def process(global_idx: int, parent: str, child: str):
        async with sem:
            try:
                rev_score, rev_resp = await _evaluate_pair_cot(
                    s_client, tokenizer, parent, child, n_samples,
                    builder, max_tokens, stop,
                )
                return global_idx, child, rev_score, rev_resp, None
            except Exception as e:
                return global_idx, child, None, [], e

    tasks = [
        asyncio.create_task(process(int(r["global_idx"]), r["parent"], r["child"]))
        for _, r in pending.iterrows()
    ]
    print(f"Dispatched {len(tasks)} pairs at concurrency={concurrency} "
          f"(after filtering {n_leaked} leaked)", flush=True)

    for completed in asyncio.as_completed(tasks):
        global_idx, child, rev_score, rev_resp, err = await completed

        if err is not None:
            print(f"  FAILED pair idx={global_idx} {child!r}: {type(err).__name__}: {err}", flush=True)
            n_failed += 1
            continue

        results.at[global_idx, rev_col]      = rev_score
        results.at[global_idx, response_col] = rev_resp[0] if rev_resp else ""
        n_done += 1

        if verbose:
            extracted = extract_answer(rev_resp[0]) if rev_resp else ""
            print(f"  [{n_done}/{len(pending)}] parent={results.at[global_idx,'parent']!r} "
                  f"-> child={child!r}: rev={rev_score:.0f}", flush=True)
            print(f"      extracted: {extracted!r}", flush=True)
            print(f"      raw[0]:    {rev_resp[0]!r}", flush=True)
        elif n_done % 25 == 0:
            elapsed = (time.time() - t0) / 60
            rate    = n_done / max(elapsed, 1e-6)
            eta     = (len(pending) - n_done) / max(rate, 1e-6)
            print(f"  done={n_done}/{len(pending)}  elapsed={elapsed:.1f}min  "
                  f"rate={rate:.1f}/min  eta={eta:.1f}min  failed={n_failed}", flush=True)

        if n_done % CHECKPOINT_EVERY == 0:
            results.to_csv(csv_path, index=False)

    return results, n_failed, n_leaked


def summarize_cot(
    results: pd.DataFrame, model: str, tag: str, n_samples: int, n_failed: int,
    n_leaked: int, condition: str, k: int | None = None,
) -> dict:
    rev_col = f"{tag}_can_find_child"
    scored  = results[results[rev_col].notna()]
    rev_mean = scored[rev_col].mean() if len(scored) else float("nan")

    summary = {
        "model":                 model,
        "condition":             condition,
        "k":                     k if condition == "fewshot_cot" else None,
        "n_pairs_total":         len(results),
        "n_pairs_leaked":        n_leaked,
        "n_pairs_scored":        len(scored),
        "n_pairs_failed":        n_failed,
        "n_samples_per_query":   n_samples,
        "reverse_accuracy_mean": round(float(rev_mean), 4) if rev_mean == rev_mean else None,
        "reverse_accuracy_pct":  round(float(rev_mean) * 100, 1) if rev_mean == rev_mean else None,
        "demo_pairs":            demo_pairs_for_k(k) if condition == "fewshot_cot" and k else None,
        "scoring_rule":          "any-of-N, contains on extracted Answer span (fallback: last non-empty line)",
        "note": f"Tinker base-model inference; condition={condition}, reverse direction only; "
                f"pairs whose names overlap with the canonical k=3 demo names are excluded "
                f"from all CoT-style runs to keep the pair set apples-to-apples.",
    }

    print(f"\n=== {model} Reversal Curse — {condition} (reverse only) ===")
    print(f"  Pairs total           : {summary['n_pairs_total']}")
    print(f"  Pairs leaked (skipped): {summary['n_pairs_leaked']}")
    print(f"  Pairs scored          : {summary['n_pairs_scored']}  (failed: {n_failed})")
    print(f"  Samples per query     : {summary['n_samples_per_query']}")
    print(f"  Reverse accuracy      : {summary['reverse_accuracy_pct']}%")
    print("=" * 56)
    return summary


def summarize(results: pd.DataFrame, model: str, tag: str, n_samples: int, n_failed: int) -> dict:
    fwd_col = f"{tag}_can_find_parent"
    rev_col = f"{tag}_can_find_child"
    fwd_mean = results[fwd_col].mean()
    rev_mean = results[rev_col].mean()

    summary = {
        "model":                 model,
        "n_pairs_total":         len(results),
        "n_pairs_scored":        int(results[fwd_col].notna().sum()),
        "n_pairs_failed":        n_failed,
        "n_samples_per_query":   n_samples,
        "forward_accuracy_mean": round(fwd_mean, 4),
        "reverse_accuracy_mean": round(rev_mean, 4),
        "reversal_gap":          round(fwd_mean - rev_mean, 4),
        "forward_accuracy_pct":  round(fwd_mean * 100, 1),
        "reverse_accuracy_pct":  round(rev_mean * 100, 1),
        "note": "Tinker base-model inference; matches Berglund et al. 2023 Experiment 2 protocol",
    }

    print(f"\n=== {model} Reversal Curse Evaluation ===")
    print(f"  Pairs scored          : {summary['n_pairs_scored']} / {summary['n_pairs_total']}  (failed: {n_failed})")
    print(f"  Samples per query     : {summary['n_samples_per_query']}")
    print(f"  Forward accuracy      : {summary['forward_accuracy_pct']}%")
    print(f"  Reverse accuracy      : {summary['reverse_accuracy_pct']}%")
    print(f"  Reversal gap          : {summary['reversal_gap']:.4f}")
    print("=" * 42)
    return summary


async def main_async(args) -> None:
    df = pd.read_csv(DATA_PATH)
    if args.max_pairs and args.max_pairs < len(df):
        df = df.sample(n=args.max_pairs, random_state=42).reset_index(drop=True)
        print(f"Sampled {len(df)} pairs (random_state=42) from {DATA_PATH}")
    else:
        print(f"Loaded {len(df)} pairs from {DATA_PATH}")
    print(f"Model: {args.model}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base_tag  = _safe_name(args.tag or args.model)
    if args.condition == "direct":
        tag = base_tag
    elif args.condition == "fewshot_cot" and args.k != K_DEFAULT:
        tag = f"{base_tag}_{args.condition}_k{args.k}"
    else:
        tag = f"{base_tag}_{args.condition}"
    csv_path  = RESULTS_DIR / f"{tag}_reversal_test_results.csv"
    json_path = RESULTS_DIR / f"{tag}_summary.json"
    print(f"Condition: {args.condition}" + (f" (k={args.k})" if args.condition == "fewshot_cot" else ""))
    print(f"Output -> {csv_path.name}, {json_path.name}")

    print("Bringing up Tinker base-model sampling client (LoRA-zero) ...")
    s_client, tokenizer = await make_base_sampling_client(args.model, name=f"{tag}_base")
    print("Sampling client ready.\n")

    if args.condition in COT_CONDITIONS:
        builder, max_tokens, stop = COT_CONDITIONS[args.condition]
        if args.condition == "fewshot_cot":
            k = args.k
            builder = lambda parent, _b=builder, _k=k: _b(parent, k=_k)
        results, n_failed, n_leaked = await run_cot_evaluation(
            df, s_client, tokenizer, tag, args.samples, csv_path, args.verbose, args.concurrency,
            builder, max_tokens, stop,
        )
        results.to_csv(csv_path, index=False)
        summary = summarize_cot(
            results, args.model, tag, args.samples, n_failed, n_leaked, args.condition,
            k=args.k if args.condition == "fewshot_cot" else None,
        )
    else:
        results, n_failed = await run_evaluation(
            df, s_client, tokenizer, tag, args.samples, csv_path, args.verbose, args.concurrency,
        )
        results.to_csv(csv_path, index=False)
        summary = summarize(results, args.model, tag, args.samples, n_failed)

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults -> {csv_path}")
    print(f"Summary -> {json_path}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Llama inference baseline on celebrity reversal task")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Tinker base model id (default: {DEFAULT_MODEL})")
    p.add_argument("--tag", default=None,
                   help="Output filename tag (default derived from --model)")
    p.add_argument("--samples", type=int, default=N_SAMPLES,
                   help=f"Samples per query, scored any-of-N (default: {N_SAMPLES})")
    p.add_argument("--max_pairs", type=int, default=None,
                   help="Cap pairs (random_state=42) — for smoke tests")
    p.add_argument("--verbose", action="store_true",
                   help="Print every pair's first sample (use for smoke test)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Number of pairs to evaluate in parallel (default: 1)")
    p.add_argument("--condition",
                   choices=["direct", "hint", "zeroshot_cot", "fewshot_cot"],
                   default="direct",
                   help="Prompt condition. 'direct' = original baseline (forward+reverse). "
                        "'hint' = single-shot reverse with a hint that the parent has a famous "
                        "child. 'zeroshot_cot' = single-shot 'let's think step by step', no "
                        "demos. 'fewshot_cot' = k-shot CoT with worked demos (k via --k). "
                        "All CoT-style conditions are reverse-only and apply the same demo-"
                        "leakage filter (against the canonical k=3 demo names) so the pair "
                        "set is apples-to-apples.")
    p.add_argument("--k", type=int, choices=[1, 2, 3, 4], default=K_DEFAULT,
                   help=f"Number of demos for --condition fewshot_cot (default: {K_DEFAULT}). "
                        f"Ignored for other conditions. k=1 keeps only Obama; k=2 drops "
                        f"Hemsworth; k=4 adds Andrea Swift -> Taylor Swift.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
