"""
Baseline 2: GPT model evaluation on real-world celebrity parent-child pairs.

Reproduces the Berglund et al. (2023) GPT-4 celebrity experiment.  For each
pair we ask:
  - Forward:  "Who is {child}'s {parent_type}?"  → expect {parent}
  - Reverse:  "Name a child of {parent}."         → expect {child}

Each question is sampled N_SAMPLES times; accuracy = fraction of responses
that start with the correct name (same scoring as original paper).

Usage (from the code/ directory, with MachineLearning conda env active):
    set OPENAI_API_KEY=sk-...
    python -m baselines.celebrity_api.evaluate [--model gpt-4o] [--samples 10] [--dry_run]

Output (model name auto-inserted):
    results/api_eval/{model}_reversal_test_results.csv
    results/api_eval/{model}_summary.json
"""

import argparse
import json
import os
import re
import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from baselines.celebrity_api.prompts import build_child_query, build_parent_query
from baselines.celebrity_api.scoring import score_responses

# ---------------------------------------------------------------------------
# Paths (relative to the code/ working directory)
# ---------------------------------------------------------------------------
DATA_PATH = "../original_repo/data/celebrity_relations/parent_child_pairs.csv"
RESULTS_DIR = "results/api_eval"

DEFAULT_MODEL    = "gpt-4o"
N_SAMPLES        = 10   # queries per direction per pair (matches original paper)
TEMPERATURE      = 1.0  # non-zero so we get varied samples; matches original
RETRY_SLEEP      = 10   # seconds to wait on rate-limit errors
CHECKPOINT_EVERY = 50   # save partial CSV every N pairs (resume-safe)


def _safe_name(model: str) -> str:
    """Filesystem-safe model tag, e.g. 'gpt-4.5-preview' stays as-is."""
    return re.sub(r"[^\w.\-]", "_", model)


def sample_completion(
    client: OpenAI,
    messages: list[dict],
    model: str,
    n: int,
    dry_run: bool,
) -> list[str | None]:
    """Call the API once with n=N_SAMPLES and return all response strings.

    If the model rejects n > some limit (e.g. gpt-5.4 caps at 8), falls back
    to making n individual calls with n=1 so the sample count stays the same.
    """
    if dry_run:
        return [None] * n

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n=n,
                temperature=TEMPERATURE,
            )
            return [choice.message.content for choice in response.choices]
        except Exception as e:
            err = str(e)
            # Model doesn't support n > some limit — use max_n in fewest calls possible
            if "integer above maximum value" in err and "'n'" in err:
                import re as _re
                m = _re.search(r"Expected a value <= (\d+)", err)
                max_n = int(m.group(1)) if m else 1
                print(f"  Model {model} caps n at {max_n}, using ceil({n}/{max_n}) calls.")
                results = []
                remaining = n
                while remaining > 0:
                    batch = min(remaining, max_n)
                    try:
                        r = client.chat.completions.create(
                            model=model, messages=messages, n=batch, temperature=TEMPERATURE,
                        )
                        results.extend(c.message.content for c in r.choices)
                    except Exception:
                        results.extend([None] * batch)
                    remaining -= batch
                return results
            if attempt == 4:
                print(f"  API error after 5 retries: {e}")
                return [None] * n
            wait = RETRY_SLEEP * (2 ** attempt)
            print(f"  API error (attempt {attempt+1}): {e} — retrying in {wait}s")
            time.sleep(wait)

    return [None] * n


def run_evaluation(
    df: pd.DataFrame,
    client: OpenAI,
    model: str,
    n_samples: int,
    dry_run: bool,
    csv_path: str,
) -> pd.DataFrame:
    """
    Evaluate all pairs with checkpoint saves every CHECKPOINT_EVERY rows.

    If csv_path already contains partial results (from a previous interrupted
    run), those rows are skipped and processing resumes from where it left off.
    """
    tag = _safe_name(model)
    fwd_col = f"{tag}_can_find_parent"
    rev_col = f"{tag}_can_find_child"

    # Build base results frame with metadata columns
    results = df[["child", "parent", "parent_type", "child_prediction"]].copy()
    results[fwd_col] = float("nan")
    results[rev_col] = float("nan")

    # Resume: load already-scored rows from a previous (partial) run
    start_idx = 0
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        if fwd_col in existing.columns:
            done = existing[fwd_col].notna().sum()
            if done > 0:
                print(f"Resuming from pair {done} / {len(df)}  ({csv_path})")
                results.loc[:done - 1, fwd_col] = existing[fwd_col].iloc[:done].values
                results.loc[:done - 1, rev_col] = existing[rev_col].iloc[:done].values
                start_idx = done

    pending = df.iloc[start_idx:].reset_index(drop=True)

    for i, (_, row) in enumerate(tqdm(pending.iterrows(), total=len(pending), desc="Evaluating pairs")):
        child, parent, parent_type = row["child"], row["parent"], row["parent_type"]
        global_idx = start_idx + i

        # Forward: child → parent
        fwd_responses = sample_completion(
            client, build_parent_query(child, parent_type), model, n_samples, dry_run
        )
        results.at[global_idx, fwd_col] = score_responses(fwd_responses, parent)

        # Reverse: parent → child
        rev_responses = sample_completion(
            client, build_child_query(parent), model, n_samples, dry_run
        )
        results.at[global_idx, rev_col] = score_responses(rev_responses, child)

        # Checkpoint every N pairs so progress survives a crash or budget cutoff
        if (i + 1) % CHECKPOINT_EVERY == 0:
            results.to_csv(csv_path, index=False)

    return results


def summarize(results: pd.DataFrame, model: str, n_samples: int) -> dict:
    tag = _safe_name(model)
    fwd_mean = results[f"{tag}_can_find_parent"].mean()
    rev_mean = results[f"{tag}_can_find_child"].mean()

    n_reversible = int(results["can_reverse"].sum()) if "can_reverse" in results.columns else "N/A"
    summary = {
        "model": model,
        "n_pairs_total": len(results),
        "n_pairs_can_reverse": n_reversible,
        "n_samples_per_query": n_samples,
        "forward_accuracy_mean": round(fwd_mean, 4),
        "reverse_accuracy_mean": round(rev_mean, 4),
        "reversal_gap": round(fwd_mean - rev_mean, 4),
        "forward_accuracy_pct": round(fwd_mean * 100, 1),
        "reverse_accuracy_pct": round(rev_mean * 100, 1),
        "note": "Tested on full 1513-pair dataset, matching Berglund et al. 2023 Experiment 2",
    }

    print(f"\n=== {model} Reversal Curse Evaluation ===")
    print(f"  Pairs evaluated       : {summary['n_pairs_total']}")
    print(f"  Samples per query     : {summary['n_samples_per_query']}")
    print(f"  Forward accuracy      : {summary['forward_accuracy_pct']}%")
    print(f"  Reverse accuracy      : {summary['reverse_accuracy_pct']}%")
    print(f"  Reversal gap          : {summary['reversal_gap']:.4f}")
    print("=" * 42 + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="GPT celebrity reversal evaluation")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"OpenAI model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--samples", type=int, default=N_SAMPLES,
                        help="Number of completions per query (default: 10)")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Cap evaluation to N randomly sampled pairs (random_state=42)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Skip API calls — useful for testing the pipeline")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    client = OpenAI(api_key=api_key)

    # Load pairs. --max_pairs caps to a random sample for quick/cheap validation runs.
    df = pd.read_csv(DATA_PATH)
    if args.max_pairs and args.max_pairs < len(df):
        df = df.sample(n=args.max_pairs, random_state=42).reset_index(drop=True)
        print(f"Sampled {len(df)} pairs (random_state=42) from {DATA_PATH}")
    else:
        print(f"Loaded {len(df)} pairs from {DATA_PATH}")
    print(f"Model: {args.model}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    tag      = _safe_name(args.model)
    csv_path = os.path.join(RESULTS_DIR, f"{tag}_reversal_test_results.csv")
    json_path = os.path.join(RESULTS_DIR, f"{tag}_summary.json")

    results = run_evaluation(df, client, args.model, args.samples, args.dry_run, csv_path)

    # Final save (also writes any rows not yet flushed by the checkpoint)
    results.to_csv(csv_path, index=False)
    summary = summarize(results, args.model, args.samples)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to: {csv_path}")
    print(f"Summary saved to: {json_path}")


if __name__ == "__main__":
    main()
